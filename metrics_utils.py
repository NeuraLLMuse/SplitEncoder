import torch
import torch.nn.functional as F
import os
import math
from bisect import bisect_right

import numpy as np
import pandas as pd
import pretty_midi
from music21 import converter, meter


def compute_normalized_token_entropy(logits, target_ids, pad_token_id=None):
    """
    Computes Expected Bits per Token (Token Entropy) for a batch.
    
    Args:
        logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size).
        target_ids (torch.Tensor): Target token IDs of shape (batch_size, seq_len).
        pad_token_id (int, optional): Token ID for padding. If provided, masked out in computation.
        
    Returns:
        entropy_per_token (torch.Tensor): Average entropy per token for each sequence.
        entropy_per_batch (float): Average entropy per token across the batch.
    """
    # Infer vocabulary size from logits shape
    vocab_size = logits.shape[-1]
    # Compute max possible entropy for normalization
    max_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32)).item()

    # Compute probabilities with softmax
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Compute log probabilities (base 2)
    log_probs = torch.log2(probs + 1e-9)  # Avoid log(0) errors

    # Compute entropy: H(x) = - sum(P(x) * log2(P(x)))
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len)

    # Mask out padding tokens if provided
    if pad_token_id is not None:
        mask = (target_ids != pad_token_id).float()  # 1 for valid tokens, 0 for padding
        entropy = entropy * mask  # Zero out entropy for padding
        entropy_per_token = entropy.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # Normalize per valid token
    else:
        entropy_per_token = entropy.mean(dim=-1)  # Average over sequence length

    # Compute overall batch entropy
    entropy_per_batch = entropy_per_token.mean().item()

    return entropy_per_token/max_entropy, entropy_per_batch/max_entropy
# end compute_token_entropy

"""MIDI harmony metrics for individual files and folders.

The metric definitions and most implementation details are retained from the
original comparison script. See compute_piece_metrics() and
compute_folder_metrics() for the simplified public API.
"""

'''
HOW TO RUN

# metrics for piece
from midi_metrics import compute_piece_metrics
metrics = compute_piece_metrics("path/to/piece.mid")
print(metrics)

# with manually defined track channels
metrics = compute_piece_metrics(
    "path/to/piece.mid",
    chord_track_index=1,
    melody_track_index=0,
    grouping_threshold=0.05,
)

# for files in folder
from midi_metrics import compute_folder_metrics
df = compute_folder_metrics("path/to/midi_folder")
print(df)

# Average metric values across the folder
print(df.drop(columns=["filename", "filepath"]).mean())

# Save the results
df.to_csv("midi_metrics.csv", index=False)

# Include pieces that got loaded with errors
df = compute_folder_metrics(
    "path/to/midi_folder",
    recursive=True,
    include_errors=True,
)

Metric	Description
CHE	Chord Histogram Entropy
CC	Chord Coverage
CTD	Chord Tonal Distance
CTnCTR	Chord Tone to Non-Chord Tone Ratio
PCS	Pitch Consonance Score
MCTD	Melody–Chord Tonal Distance
HRHE_i	Harmonic Rhythm Histogram Entropy, based on inter-chord intervals
HRC_i	Harmonic Rhythm Coverage, based on inter-chord intervals
CBS	Chord Beat Strength

'''

def _get_bar_segments(pm: pretty_midi.PrettyMIDI):
    """
    Return a list of (bar_start_time, bar_end_time) in seconds.
    Prefers PrettyMIDI downbeats; falls back to 4/4 (every 4 beats).
    """
    end_time = pm.get_end_time()
    bar_starts = []

    downbeats = pm.get_downbeats()
    if len(downbeats) >= 1:
        bar_starts = list(downbeats)
        if bar_starts[0] > 0.0:
            bar_starts = [0.0] + bar_starts
    else:
        beats = pm.get_beats()
        if len(beats) >= 1:
            bar_starts = beats[::4]  # assume 4/4
            if not bar_starts or bar_starts[0] > 0.0:
                bar_starts = [0.0] + bar_starts
        else:
            bar_starts = [0.0]

    bars = []
    for i, s in enumerate(bar_starts):
        e = bar_starts[i+1] if i+1 < len(bar_starts) else end_time
        if e > s:
            bars.append((s, e))
    return bars

def _build_halfbar_table(pm: pretty_midi.PrettyMIDI, chords_with_times):
    """
    Build half-bar segments with an assigned chord:
      returns list of (half_start, half_end, chord_or_None)
    The chord for a half-bar is the chord active at half_start:
    last chord whose onset <= half_start; if none, chord is None.
    """
    bars = _get_bar_segments(pm)
    halfbars = []
    chord_times = [t for t, _ in chords_with_times]
    chord_vals  = [c for _, c in chords_with_times]

    for (s, e) in bars:
        mid = 0.5 * (s + e)
        for hs, he in ((s, mid), (mid, e)):
            idx = bisect_right(chord_times, hs) - 1
            chord = chord_vals[idx] if idx >= 0 else None
            halfbars.append((hs, he, chord))
    return halfbars

def _chord_from_halfbar(halfbar_table, t):
    """
    Return the chord assigned to the half-bar that contains time t.
    If no half-bar contains t (rare at the very end), return None.
    """
    starts = [hs for (hs, _, _) in halfbar_table]
    idx = bisect_right(starts, t) - 1
    if idx >= 0:
        hs, he, chord = halfbar_table[idx]
        if hs <= t < he:
            return chord
    return None

def _sixteenth_windows(pm: pretty_midi.PrettyMIDI):
    """
    Build musical sixteenth-note windows in seconds.
    Prefer beat-subdivision (each beat -> 4 sixteenths). If beats insufficient,
    fall back to uniform (60/BPM)/4 seconds using median tempo (or 120 BPM).
    """
    windows = []
    beats = pm.get_beats()
    end_time = pm.get_end_time()

    if len(beats) >= 2:
        # Subdivide all complete beat intervals
        for i in range(len(beats) - 1):
            b0, b1 = beats[i], beats[i+1]
            step = (b1 - b0) / 4.0
            for j in range(4):
                windows.append((b0 + j * step, b0 + (j + 1) * step))
        # Tail: from last beat to end_time
        last = beats[-1]
        if end_time > last:
            # approximate last step by previous beat length (or leave as one window)
            prev_len = beats[-1] - beats[-2] if len(beats) >= 2 else (end_time - last)
            step = prev_len / 4.0 if prev_len > 0 else (end_time - last)
            t = last
            while t < end_time - 1e-9:
                windows.append((t, min(t + step, end_time)))
                t += step
    else:
        tempi_times, tempi_vals = pm.get_tempo_changes()
        bpm = float(np.median(tempi_vals)) if len(tempi_vals) > 0 else 120.0
        sixteenth = (60.0 / bpm) / 4.0
        t = 0.0
        while t < end_time - 1e-9:
            windows.append((t, min(t + sixteenth, end_time)))
            t += sixteenth
    return windows

def extract_chords_with_times_from_midi(midi_file_path, track_index=1, grouping_threshold=0.05):
    """
    Extract chords from the specified track of a MIDI file along with their onset times.
    
    Parameters:
        midi_file_path (str): Path to the MIDI file.
        track_index (int): Track index for chords.
        grouping_threshold (float): Time threshold to group notes.
    
    Returns:
        List[Tuple[float, Tuple[int]]]: A list of (onset_time, chord) pairs.
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    if track_index >= len(pm.instruments):
        raise ValueError(f"Track index {track_index} is out of range for file {midi_file_path}.")
    
    chord_instrument = pm.instruments[track_index]
    sorted_notes = sorted(chord_instrument.notes, key=lambda note: note.start)
    
    chords_with_times = []
    current_chord_notes = []
    current_time = None

    for note in sorted_notes:
        if current_time is None:
            current_time = note.start
            current_chord_notes.append(note)
        else:
            if note.start - current_time <= grouping_threshold:
                current_chord_notes.append(note)
            else:
                chord_pitch_classes = tuple(sorted({n.pitch % 12 for n in current_chord_notes}))
                chords_with_times.append((current_time, chord_pitch_classes))
                current_chord_notes = [note]
                current_time = note.start

    if current_chord_notes:
        chord_pitch_classes = tuple(sorted({n.pitch % 12 for n in current_chord_notes}))
        chords_with_times.append((current_time, chord_pitch_classes))
    
    return chords_with_times

def tonal_centroid(notes):
    """
    Compute the 6-D tonal centroid for a set of pitch classes using lookup tables.
    The centroid is computed by averaging three 2-D vectors:
      - Fifths (e.g. a perfect fifth is assigned [1.0, 0.0])
      - Minor thirds
      - Major thirds
    The three 2-D vectors are concatenated to form a 6-D vector.
    
    Parameters:
        notes (list of int): List of pitch class numbers (0-11).
    
    Returns:
        np.ndarray: A 6-D numpy array representing the tonal centroid.
    """
    # Lookup dictionaries:
    fifths_lookup = {
        9: [1.0, 0.0],
        2: [math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)],
        7: [math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
        0: [0.0, 1.0],
        5: [math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)],
        10: [math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
        3: [-1.0, 0.0],
        8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        1: [math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
        6: [0.0, -1.0],
        11: [math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)],
        4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]
    }
    minor_thirds_lookup = {
        3: [1.0, 0.0],
        7: [1.0, 0.0],
        11: [1.0, 0.0],
        0: [0.0, 1.0],
        4: [0.0, 1.0],
        8: [0.0, 1.0],
        1: [-1.0, 0.0],
        5: [-1.0, 0.0],
        9: [-1.0, 0.0],
        2: [0.0, -1.0],
        6: [0.0, -1.0],
        10: [0.0, -1.0]
    }
    major_thirds_lookup = {
        0: [0.0, 1.0],
        3: [0.0, 1.0],
        6: [0.0, 1.0],
        9: [0.0, 1.0],
        2: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        5: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        11: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
        1: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
        4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
        7: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
        10: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]
    }
    
    # Weights for each component:
    r1 = 1.0  # for fifths
    r2 = 1.0  # for minor thirds
    r3 = 0.5  # for major thirds
    
    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    
    if notes:
        for note in notes:
            for i in range(2):
                fifths[i] += r1 * fifths_lookup[note][i]
                minor[i] += r2 * minor_thirds_lookup[note][i]
                major[i] += r3 * major_thirds_lookup[note][i]
        n = len(notes)
        for i in range(2):
            fifths[i] /= n
            minor[i] /= n
            major[i] /= n
    return np.array(fifths + minor + major)

def compute_chord_histogram_entropy(chords):
    """
    Compute the Chord Histogram Entropy (CHE) of a chord sequence.
    
    Parameters:
        chords (List[Tuple[int]]): List of chords (each chord is a tuple of pitch classes).
    
    Returns:
        float: Entropy in bits.
    """
    if not chords:
        return 0.0
    histogram = {}
    for chord in chords:
        histogram[chord] = histogram.get(chord, 0) + 1
    total = len(chords)
    entropy = 0.0
    for count in histogram.values():
        p = count / total
        entropy -= p * np.log(p + 1e-6)
    return entropy

def compute_chord_coverage(chords):
    """
    Compute the Chord Coverage (CC) of a chord sequence.
    
    Parameters:
        chords (List[Tuple[int]]): List of chords.
    
    Returns:
        int: Number of unique chords.
    """
    return len(set(chords))

def chord_to_tonal_vector(chord):
    """
    Compute the 6-D tonal vector (tonal centroid) for a chord.
    
    Parameters:
        chord (Tuple[int]): A chord represented as a tuple of pitch classes.
    
    Returns:
        np.ndarray: 6-D tonal vector.
    """
    return tonal_centroid(list(chord))

def chord_tonal_distance(chord1, chord2):
    """
    Compute Euclidean distance between the tonal vectors of two chords.
    
    Parameters:
        chord1, chord2 (Tuple[int]): Chords as tuples of pitch classes.
    
    Returns:
        float: Euclidean distance.
    """
    tv1 = chord_to_tonal_vector(chord1)
    tv2 = chord_to_tonal_vector(chord2)
    return np.linalg.norm(tv1 - tv2)

def compute_chord_tonal_distance(chords):
    """
    Compute average Chord Tonal Distance (CTD) over a sequence of chords.
    
    Parameters:
        chords (List[Tuple[int]]): List of chords.
    
    Returns:
        float: Average CTD.
    """
    if len(chords) < 2:
        return 0.0
    distances = []
    for i in range(1, len(chords)):
        distances.append(chord_tonal_distance(chords[i-1], chords[i]))
    return np.mean(distances)

def compute_ctnctr(midi_file_path, chord_track_index=1, melody_track_index=0, grouping_threshold=0.05):
    """
    CTnCTR aligned with paper:
      - Active chord is the chord label of the *half-bar* containing the melody note.
      - Proper NCT (n_p): an NCT whose immediately following different note is
        a chord tone of the *same* half-bar chord and within ≤ 2 semitones.
      - Score: (n_c + n_p) / (n_c + n_n)
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    if melody_track_index >= len(pm.instruments):
        raise ValueError(f"Melody track index {melody_track_index} out of range for file {midi_file_path}.")

    melody_notes = sorted(pm.instruments[melody_track_index].notes, key=lambda note: note.start)
    chords_with_times = extract_chords_with_times_from_midi(
        midi_file_path, track_index=chord_track_index, grouping_threshold=grouping_threshold
    )
    if not chords_with_times:
        return 0.0

    halfbars = _build_halfbar_table(pm, chords_with_times)

    nc = 0  # chord tones
    nn = 0  # non-chord tones
    np_ = 0 # proper NCTs

    for idx, note in enumerate(melody_notes):
        note_pc = note.pitch % 12
        chord = _chord_from_halfbar(halfbars, note.start)
        if not chord:  # N.C. for this half-bar
            nn += 1
            continue

        if note_pc in chord:
            nc += 1
        else:
            nn += 1
            # proper NCT test against the *same half-bar chord*
            for next_note in melody_notes[idx+1:]:
                if next_note.pitch != note.pitch:
                    if (next_note.pitch % 12) in chord and abs(next_note.pitch - note.pitch) <= 2:
                        np_ += 1
                    break

    denom = nc + nn
    return (nc + np_) / denom if denom > 0 else 0.0

def compute_pcs(midi_file_path, chord_track_index=1, melody_track_index=0, grouping_threshold=0.05):
    """
    PCS aligned with paper:
      - For each melody note, compare to the chord of the *half-bar* containing the note.
      - Force chord tones to the octave at-or-below the melody pitch.
      - Score per interval: +1 for {0,3,4,7,8,9}, 0 for 5, -1 otherwise.
      - Average over 1/16 windows (musical), excluding windows with no notes.
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    chords_with_times = extract_chords_with_times_from_midi(
        midi_file_path, track_index=chord_track_index, grouping_threshold=grouping_threshold
    )
    if not chords_with_times:
        return 0.0

    if melody_track_index >= len(pm.instruments):
        raise ValueError(f"Melody track index {melody_track_index} out of range for file {midi_file_path}.")
    melody_notes = sorted(pm.instruments[melody_track_index].notes, key=lambda note: note.start)

    halfbars = _build_halfbar_table(pm, chords_with_times)
    windows = _sixteenth_windows(pm)

    def consonance_score(interval):
        if interval in {0, 3, 4, 7, 8, 9}:
            return 1
        elif interval == 5:
            return 0
        return -1

    window_scores = []
    for (w_start, w_end) in windows:
        notes_in_window = [n for n in melody_notes if w_start <= n.start < w_end]
        if not notes_in_window:
            continue
        note_scores = []
        for note in notes_in_window:
            chord = _chord_from_halfbar(halfbars, note.start)
            if not chord:
                # N.C. → treat as non-chord: consonance undefined; count as -1 across the board?
                # Paper excludes rests; for N.C. we'll just skip scoring this note.
                # (Alternative would be to score as -1; we choose skip to avoid bias.)
                continue
            chord_tone_scores = []
            for chord_pc in chord:
                k = (note.pitch - chord_pc) // 12
                chord_pitch = chord_pc + 12 * k
                if chord_pitch > note.pitch:
                    chord_pitch -= 12
                interval = note.pitch - chord_pitch
                chord_tone_scores.append(consonance_score(interval))
            if chord_tone_scores:
                note_scores.append(float(np.mean(chord_tone_scores)))
        if note_scores:
            window_scores.append(float(np.mean(note_scores)))

    return float(np.mean(window_scores)) if window_scores else 0.0

def compute_mctd(midi_file_path, chord_track_index=1, melody_track_index=0, grouping_threshold=0.05):
    """
    MCTD aligned with paper:
      - Melody note: singleton PCP → 6-D tonal centroid (Harte).
      - Chord: half-bar chord label → PCs → 6-D centroid.
      - Distance: Euclidean; duration-weighted average over notes.
      - If the half-bar chord is N.C., skip that note (undefined reference).
    """
    pm = pretty_midi.PrettyMIDI(midi_file_path)
    if melody_track_index >= len(pm.instruments):
        raise ValueError(f"Melody track index {melody_track_index} out of range for file {midi_file_path}.")
    melody_notes = sorted(pm.instruments[melody_track_index].notes, key=lambda note: note.start)

    chords_with_times = extract_chords_with_times_from_midi(
        midi_file_path, track_index=chord_track_index, grouping_threshold=grouping_threshold
    )
    if not chords_with_times:
        return 0.0

    halfbars = _build_halfbar_table(pm, chords_with_times)

    weighted_sum = 0.0
    total_dur = 0.0

    for note in melody_notes:
        chord = _chord_from_halfbar(halfbars, note.start)
        if not chord:
            continue  # N.C. half-bar → undefined distance; skip
        melody_tc = tonal_centroid([note.pitch % 12])
        chord_tc  = tonal_centroid(list(chord))
        d = float(np.linalg.norm(melody_tc - chord_tc))
        dur = note.end - note.start
        weighted_sum += d * dur
        total_dur += dur

    return (weighted_sum / total_dur) if total_dur > 0 else 0.0

def compute_HRHE_and_HRC_intervals(chords_with_times, quantization=0.05):
    """
    Compute the Harmonic Rhythm Histogram Entropy (HRHE) and Harmonic Rhythm Coverage (HRC)
    from a list of chord events with onset times.
    
    Parameters:
        chords_with_times (List[Tuple[float, chord]]): List of tuples where each tuple contains
            the onset time (in seconds) and a chord (represented as pitch classes).
        quantization (float): Quantization step (in seconds) used to discretize the inter-chord intervals.
    
    Returns:
        tuple: (HRHE, HRC)
            HRHE (float): The entropy (in nats) of the histogram of quantized inter-chord intervals.
            HRC (int): The number of unique quantized interval types.
    """
    if len(chords_with_times) < 2:
        return 0.0, 0

    # Compute inter-onset intervals between successive chord events.
    intervals = []
    for i in range(1, len(chords_with_times)):
        diff = chords_with_times[i][0] - chords_with_times[i-1][0]
        # Quantize the interval (e.g., rounding to the nearest multiple of 'quantization')
        quantized_diff = round(diff / quantization) * quantization
        intervals.append(quantized_diff)
    # print(intervals)
    
    # Build a histogram (frequency count) of the quantized intervals.
    hist = {}
    for diff in intervals:
        hist[diff] = hist.get(diff, 0) + 1

    total = sum(hist.values())
    HRHE = 0.0
    for count in hist.values():
        p = count / total
        HRHE -= p * np.log(p + 1e-6)  # using a small epsilon to avoid log(0)
    
    # HRC: number of unique quantized interval types.
    HRC = len(hist)
    return HRHE, HRC

def compute_CBS(midi_file_path, chord_part_index=1, tolerance=0.05):
    """
    Compute the Chord Beat Strength (CBS) for a MIDI file.
    
    This implementation uses music21 to extract measures and detect the time signature 
    for each measure. For each chord event (extracted by chordifying each measure), the 
    onset is measured relative to the start of the measure (in quarter-note units).
    
    The ideal positions are defined on a 16th-note grid:
      - For a measure with time signature A/B:
          • Beat duration (in quarter notes) is: beat_duration = 4 / B.
          • The number of beats in the measure is A.
      - Strong beats are at offsets: 0, 1×beat_duration, 2×beat_duration, …, (A-1)×beat_duration.
      - Half-beats are at: (i + 0.5)×beat_duration for i = 0,1,...,A-1.
      - Eighth-note subdivisions are at: (i + 0.25)×beat_duration and (i + 0.75)×beat_duration.
    
    Scoring is defined as follows (using a 4/4 example):
      - Score 0 if the chord is at the beginning of the measure (offset 0),
      - Score 1 if the chord is on any other strong beat (e.g., 1.0, 2.0, or 3.0),
      - Score 2 if the chord is on a half-beat (e.g., 0.5, 1.5, etc.),
      - Score 3 if the chord is on an eighth-note subdivision (e.g., 0.25, 0.75, etc.),
      - Score 4 for all other positions.
    
    The final CBS is the average of these scores over all chord events.
    
    Parameters:
        midi_file_path (str): Path to the MIDI file.
        chord_part_index (int): Index of the part containing chords.
        tolerance (float): Tolerance (in quarter notes) to consider an event "on" an ideal position.
    
    Returns:
        float: The average CBS score.
    """
    # Parse the MIDI file using music21.
    score = converter.parse(midi_file_path)
    
    # Select the chord part.
    try:
        chord_part = score.parts[chord_part_index]
    except IndexError:
        raise ValueError(f"Chord part index {chord_part_index} is out of range.")
    
    # Extract measures from the chord part.
    measures = chord_part.getElementsByClass('Measure')
    if not measures:
        raise ValueError("No measures found in the chord part.")
    
    def ideal_positions_for_measure(ts):
        """
        Given a TimeSignature object, return three lists:
          - strong: positions (in quarter notes) of strong beats,
          - half: positions for half-beats,
          - eighth: positions for eighth-note subdivisions.
        """
        num_beats = ts.numerator
        beat_duration = 4 / ts.denominator  # Duration of one beat in quarter notes.
        measure_duration = num_beats * beat_duration
        
        strong = [i * beat_duration for i in range(num_beats)]
        half = [i * beat_duration + 0.5 * beat_duration for i in range(num_beats)]
        
        eighth = []
        for i in range(num_beats):
            pos1 = i * beat_duration + 0.25 * beat_duration
            pos2 = i * beat_duration + 0.75 * beat_duration
            if pos1 < measure_duration:
                eighth.append(pos1)
            if pos2 < measure_duration:
                eighth.append(pos2)
        return strong, half, eighth
    
    scores = []
    for m in measures:
        # Get the time signature for the measure. Use the first if there are multiple.
        ts_list = m.getTimeSignatures()
        ts = ts_list[0] if ts_list else meter.TimeSignature('4/4')
        
        strong, half, eighth = ideal_positions_for_measure(ts)
        
        # Chordify the measure to collapse simultaneous events into chord events.
        m_chords = m.chordify().recurse().getElementsByClass('Chord')
        for ch in m_chords:
            offset = ch.offset  # offset within the measure (in quarter notes)
            # New scoring: 
            # 0 if exactly at beginning of measure, 1 if on other strong beats,
            # 2 if on half-beat, 3 if on eighth subdivision, 4 otherwise.
            if abs(offset - strong[0]) <= tolerance:
                score_val = 0
            elif any(abs(offset - pos) <= tolerance for pos in strong[1:]):
                score_val = 1
            elif any(abs(offset - pos) <= tolerance for pos in half):
                score_val = 2
            elif any(abs(offset - pos) <= tolerance for pos in eighth):
                score_val = 3
            else:
                score_val = 4
            scores.append(score_val)
    
    return np.mean(scores) if scores else 0.0

def compute_piece_metrics(
    midi_file_path,
    chord_track_index=1,
    melody_track_index=0,
    grouping_threshold=0.05,
):
    """Compute all supported metrics for one MIDI file.

    Parameters
    ----------
    midi_file_path : str or path-like
        Path to a MIDI file. By default, track 0 is melody and track 1 chords.
    chord_track_index, melody_track_index : int
        Instrument indices in the MIDI file.
    grouping_threshold : float
        Notes beginning within this many seconds are grouped as one chord.
        For compatibility with the original script, this value is also passed
        as the quantization for harmonic-rhythm intervals and the CBS tolerance.

    Returns
    -------
    dict
        CHE, CC, CTD, HRHE_i, HRC_i, CBS, CTnCTR, PCS, and MCTD.
    """
    midi_file_path = os.fspath(midi_file_path)
    chords_with_times = extract_chords_with_times_from_midi(
        midi_file_path,
        track_index=chord_track_index,
        grouping_threshold=grouping_threshold,
    )
    chords = [chord for _, chord in chords_with_times]

    metrics = {
        "CHE": compute_chord_histogram_entropy(chords),
        "CC": compute_chord_coverage(chords),
        "CTD": compute_chord_tonal_distance(chords),
    }
    metrics["HRHE_i"], metrics["HRC_i"] = compute_HRHE_and_HRC_intervals(
        chords_with_times, quantization=grouping_threshold
    )
    metrics["CBS"] = compute_CBS(
        midi_file_path, chord_part_index=chord_track_index,
        tolerance=grouping_threshold
    )
    metrics["CTnCTR"] = compute_ctnctr(
        midi_file_path, chord_track_index, melody_track_index, grouping_threshold
    )
    metrics["PCS"] = compute_pcs(
        midi_file_path, chord_track_index, melody_track_index, grouping_threshold
    )
    metrics["MCTD"] = compute_mctd(
        midi_file_path, chord_track_index, melody_track_index, grouping_threshold
    )
    return metrics


def compute_folder_metrics(
    folder_path,
    chord_track_index=1,
    melody_track_index=0,
    grouping_threshold=0.05,
    recursive=False,
    include_errors=False,
):
    """Compute metrics for every MIDI file in a folder and return a DataFrame.

    Files are processed in sorted order. By default, invalid files are skipped
    with a warning. Set ``include_errors=True`` to include a row for each failed
    file, with its error message in an ``error`` column and NaN metric values.
    Set ``recursive=True`` to include MIDI files in subfolders.

    The returned DataFrame contains ``filename``, ``filepath``, and one column
    per metric (plus ``error`` if requested).
    """
    folder_path = os.fspath(folder_path)
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    if recursive:
        midi_paths = [
            os.path.join(root, name)
            for root, _, names in os.walk(folder_path)
            for name in names
            if name.lower().endswith((".mid", ".midi"))
        ]
    else:
        midi_paths = [
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.lower().endswith((".mid", ".midi"))
            and os.path.isfile(os.path.join(folder_path, name))
        ]
    midi_paths.sort()

    metric_names = ["CHE", "CC", "CTD", "HRHE_i", "HRC_i", "CBS", "CTnCTR", "PCS", "MCTD"]
    rows = []
    for midi_path in midi_paths:
        base = {
            "filename": os.path.basename(midi_path),
            "filepath": midi_path,
        }
        try:
            base.update(compute_piece_metrics(
                midi_path,
                chord_track_index=chord_track_index,
                melody_track_index=melody_track_index,
                grouping_threshold=grouping_threshold,
            ))
            if include_errors:
                base["error"] = None
            rows.append(base)
        except Exception as exc:
            message = f"Skipping {midi_path}: {exc}"
            if include_errors:
                base.update({metric: np.nan for metric in metric_names})
                base["error"] = str(exc)
                rows.append(base)
            else:
                print(message)

    columns = ["filename", "filepath"] + metric_names
    if include_errors:
        columns.append("error")
    return pd.DataFrame(rows, columns=columns)


# Backwards-compatible name for code that used the old single-file pipeline.
compute_all_metrics = compute_piece_metrics
