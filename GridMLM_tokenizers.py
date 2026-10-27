from logging import root

from tqdm import tqdm
from transformers import PreTrainedTokenizer
from music21 import converter, harmony, pitch, note, interval, stream, meter, chord, duration
import mir_eval
from copy import deepcopy
import numpy as np
import os
import json
import ast
from copy import deepcopy
import random
from music_utils import detect_key, get_transposition_interval, transpose_score
import torch

INT_TO_ROOT_SHARP = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}


MIR_QUALITIES = mir_eval.chord.QUALITIES
EXT_MIR_QUALITIES = deepcopy( MIR_QUALITIES )
CHORD_FEATURES = {}
for k in list(MIR_QUALITIES.keys()) + ['7(b9)', '7(#9)', '7(#11)', '7(b13)']:
    _, semitone_bitmap, _ = mir_eval.chord.encode( 'C' + (len(k) > 0)*':' + k, reduce_extended_chords=True )
    EXT_MIR_QUALITIES[k] = semitone_bitmap
    for r_str in INT_TO_ROOT_SHARP.values():
        # rt, semitone_bitmap, _ = mir_eval.chord.encode( r_str + (len(k) > 0)*':' + k, reduce_extended_chords=True )
        rt, _, _ = mir_eval.chord.encode( r_str + (len(k) > 0)*':' + k, reduce_extended_chords=True )
        bin_pcs = np.roll(semitone_bitmap, rt)
        CHORD_FEATURES[ r_str + (len(k) > 0)*':' + k ] = {
            'quality': k,
            'root': rt,
            'pitch_classes': np.where(bin_pcs)[0].tolist()
        }

class CSGridMLMTokenizer(PreTrainedTokenizer):
    def __init__(
            self,
            quantization='16th',
            fixed_length=None,
            vocab=None,
            special_tokens=None,
            use_pc_roll=True,
            use_full_range_melody=True,
            intertwine_bar_info=True,
            trim_start=False,
            **kwargs
        ):
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.no_chord = '<nc>'
        self.nc_token = '<nc>'
        self.csl_token = '<s>'
        self.mask_token = '<mask>'
        self.bar_token = '<bar>'
        self.FIST_CHORD_TOKEN_INDEX = 7
        self.quantization = quantization
        self.fixed_length = fixed_length
        self.intertwine_bar_info = intertwine_bar_info
        self.trim_start = trim_start
        self.special_tokens = {}
        self.use_pc_roll = use_pc_roll
        self.use_full_range_melody = use_full_range_melody
        self.pianoroll_dim = 88*use_full_range_melody + 12*use_pc_roll + intertwine_bar_info
        self.construct_basic_vocab()
        if vocab is not None:
            self.vocab = vocab
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self._added_tokens_encoder = {}
        else:
            self.special_tokens = {} # not really needed in this implementation
            self._added_tokens_encoder = {} # TODO: allow for special tokens
        chromatic_roots = []
        for i in range(12):
            pitch_obj = pitch.Pitch(i)
            # Convert flat notation to sharp
            if '-' in pitch_obj.name:  # Check for flats
                pitch_obj = pitch_obj.getEnharmonic()  # Convert to sharp
            chromatic_roots.append(pitch_obj.name)  # Use sharp representation

        qualities = list(EXT_MIR_QUALITIES.keys())

        for root in chromatic_roots:
            for quality in qualities:
                    chord_token = root + (len(quality) > 0)*':' + quality
                    self.vocab[chord_token] = len(self.vocab)
        self.update_ids_to_tokens()
        self.total_vocab_size = len(self.vocab)
    # end init

    def construct_basic_vocab(self):
        self.vocab = {
                '<unk>': 0,
                '<pad>': 1,
                '<s>': 2,
                '</s>': 3,
                '<nc>': 4,
                '<mask>': 5,
                '<bar>': 6
            }

        self.update_ids_to_tokens()
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.nc_token_id = 4
        self.mask_token_id = 5
        self.bar_token_id = 6
        # Compute and store most popular time signatures coming from predefined time tokens
        self.time_quantization = []  # Store predefined quantized times
        self.time_signatures = []  # Store most common time signatures
        # Predefine time quantization tokens
        max_quarters = 10  # Support up to 10/4 time signatures
        subdivisions = [0, 0.16, 0.25, 0.33, 0.5, 0.66, 0.75, 0.83]
        for quarter in range(max_quarters):
            for subdivision in subdivisions:
                quant_time = round(quarter + subdivision, 3)
                self.time_quantization.append(quant_time)
        self.time_signatures = self.infer_time_signatures_from_quantization(self.time_quantization, max_quarters)
    # end construct_basic_vocab

    def infer_time_signatures_from_quantization(self, time_quantization, max_quarters=10):
        """
        Calculate time signatures based on the quantization scheme. Only x/4 and x/8 are
        included. Removing duplicates like 2/4 and 4/8 keeping the simplest denominator.
        """
        inferred_time_signatures = set()

        for measure_length in range(1, max_quarters + 1):
            # Extract tokens within the current measure
            measure_tokens = [t for t in time_quantization if int(t) < measure_length]

            # Add `x/4` time signatures (number of quarters in the measure)
            inferred_time_signatures.add((measure_length, 4))

            # Validate all valid groupings for `x/8`
            for numerator in range(1, measure_length * 2 + 1):  # Up to 2 eighths per quarter
                eighth_duration = 0.5  # Fixed duration for eighth notes
                valid_onsets = [i * eighth_duration for i in range(numerator)]
                
                # Check if measure_tokens contains a valid subset matching the onsets
                if all(any(abs(t - onset) < 0.01 for t in measure_tokens) for onset in valid_onsets):
                    inferred_time_signatures.add((numerator, 8))
        
        # Remove equivalent time signatures. Separate x/4 and x/8 time signatures
        quarter_signatures = {num for num, denom in inferred_time_signatures if denom == 4}
        cleaned_signatures = [] 
        
        for num, denom in inferred_time_signatures:
            # Keep x/4 time signatures
            if denom == 4:
                cleaned_signatures.append((num, denom))
            # Keep x/8 only if there's no equivalent x/4
            elif denom == 8 and num / 2 not in quarter_signatures:
                cleaned_signatures.append((num, denom))              

        # Return sorted time signatures
        return sorted(cleaned_signatures)
    # end infer_time_signatures_from_quantization

    def update_ids_to_tokens(self):
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    # end update_ids_to_tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab[tokens]
        return [self.vocab[token] for token in tokens]
    # end convert_tokens_to_ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.unk_token)
        return [self.ids_to_tokens[i] for i in ids]
    # end convert_ids_to_tokens
    
    def normalize_root_to_sharps(self, root):
        """
        Normalize chord roots to sharp notation, handling special cases like '-' for sharps.
        """
        # Custom mapping for cases like "D-" → "C#"
        special_mapping = {
            'C-': 'B',
            'D-': 'C#',
            'E-': 'D#',
            'F-': 'E',
            'E#': 'F',
            'G-': 'F#',
            'A-': 'G#',
            'B-': 'A#',
            'B#': 'C',
            'C##': 'D',
            'D##': 'E',
            'E##': 'F#',
            'F##': 'G',
            'G##': 'A',
            'A##': 'B',
            'B##': 'C#',
            'C--': 'A#',
            'D--': 'C',
            'E--': 'D',
            'F--': 'D#',
            'G--': 'F',
            'A--': 'G',
            'B--': 'A'
        }

        # Check if the root matches a special case
        if root in special_mapping:
            return special_mapping[root]

        # Use music21 to normalize root to sharp notation otherwise
        pitch_obj = pitch.Pitch(root)
        return pitch_obj.name  # Always return the sharp representation
    # end normalize_root_to_sharps

    def get_closest_mir_eval_symbol(self, chord_symbol):
        # get binary type representation
        # transpose to c major
        ti = interval.Interval( chord_symbol.root(), pitch.Pitch('C') )
        tc = chord_symbol.transpose(ti)
        # make binary
        b = np.zeros(12)
        b[tc.pitchClasses] = 1
        similarity_max = -1
        key_max = '<unk>'
        for k in EXT_MIR_QUALITIES.keys():
            tmp_similarity = np.sum(b == EXT_MIR_QUALITIES[k])
            if similarity_max < tmp_similarity:
                similarity_max = tmp_similarity
                key_max = k
        return key_max
    # end get_closest_mir_eval_symbol

    def normalize_chord_symbol(self, chord_symbol):
        """
        Normalize a music21 chord symbol to match the predefined vocabulary.
        """
        # Normalize root to sharp notation
        root = self.normalize_root_to_sharps(chord_symbol.root().name)  # E.g., "Db" → "C#"
        quality = self.get_closest_mir_eval_symbol( chord_symbol )
        # Return the normalized chord symbol
        return f"{root}", f"{quality}"
    # end normalize_chord_symbol

    def handle_chord_symbol(self, h):
        # from chord symbol tokenizer for the time being
        # Normalize and add the chord symbol
        root_token, type_token = self.normalize_chord_symbol(h)
        chord_token = root_token + (len(type_token) > 0)*':' + type_token
        if chord_token in self.vocab:
            chord_token_id = self.vocab[chord_token]
        else:
            # Handle unknown chords
            chord_token = self.unk_token
            chord_token_id = self.vocab[self.unk_token]
        return chord_token, chord_token_id
    # end handle_chord_symbol

    def decode_chord_symbol(self, harmony_tokens):
        raise NotImplementedError()
    # end decode_chord_symbol

    def fit(self, corpus):
        pass
    # end fit

    def transform(self, corpus, add_start_harmony_token=True):
        tokens = []
        ids = []
        for file_path in tqdm(corpus, desc="Processing Files"):
            encoded = self.encode(file_path, add_start_harmony_token=add_start_harmony_token)
            harmony_tokens = encoded['harmony_tokens']
            harmony_ids = encoded['harmony_ids']
            tokens.append(harmony_tokens)
            ids.append(harmony_ids)
        return {'tokens': tokens, 'ids': ids}
    # end transform

    def randomize_score(self, score, note_remove_pct=0., chord_remove_pct=0., note_change_pct=0.):
        """
        Modifies a music21 score by:
        1. Removing a percentage of melody notes.
        2. Removing a percentage of harmony chords.
        3. Shifting a percentage of melody notes by a given number of semitones.
        
        Parameters:
        - score (music21.stream.Score): The input score.
        - note_remove_pct (float): Percentage of melody notes to remove.
        - chord_remove_pct (float): Percentage of harmony chords to remove.
        - note_change_pct (float): Percentage of melody notes to shift.
        - shift_semitones (int): Number of semitones to shift the notes.
        
        Returns:
        - Modified music21.stream.Score
        """
        # Get the first part
        part = score.parts[0]
        
        # Separate notes and chord symbols
        notes = [n for n in part.notes if isinstance(n, note.Note)]
        chords = [c for c in part.notes if isinstance(c, harmony.ChordSymbol)]
        
        # Remove random notes
        num_notes_remove = int(len(notes) * note_remove_pct)
        notes_to_remove = random.sample(notes, num_notes_remove)
        
        for note in notes_to_remove:
            part.remove(note)
        
        # Remove random chord symbols
        num_chords_remove = int(len(chords) * chord_remove_pct)
        chords_to_remove = random.sample(chords, num_chords_remove)
        
        for chord in chords_to_remove:
            part.remove(chord)
        
        # Shift random notes by n semitones
        num_notes_change = int(len(notes) * note_change_pct/2)
        notes_to_change = random.sample(notes, num_notes_change)
        
        for note in notes_to_change:
            shift_semitones = np.random.randint(-3,3)
            note.transpose(shift_semitones, inPlace=True)
        
        return score
    # end randomize_score

    def pitch_class_from_chord_token(self, chord_token):
        """
        Convert a chord token to its pitch class profile.
        
        Parameters
        ----------
        chord_token : str
            Chord token in the format "Root:Quality" or "<nc>" or "<pad>".
        
        Returns
        -------
        np.ndarray
            12D binary vector representing the pitch class profile.
        """
        if chord_token in [self.nc_token, self.pad_token, self.bar_token]:
            return np.zeros(12, dtype=int)

        if chord_token not in self.vocab:
            return np.zeros(12, dtype=int)

        try:
            root, quality = chord_token.split(':')
        except ValueError:
            root = chord_token
            quality = ''
        
        if quality not in EXT_MIR_QUALITIES:
            return np.zeros(12, dtype=int)

        # Get root pitch class
        root_pitch = pitch.Pitch(root)
        root_pc = root_pitch.pitchClass

        # Get quality pitch class bitmap
        quality_bitmap = EXT_MIR_QUALITIES[quality]

        # Rotate bitmap to match root
        pc_profile = np.roll(quality_bitmap, root_pc)

        return pc_profile
    # end pitch_class_from_chord_token

    def to_category(self, x, thresholds):
        if x <= thresholds[0]:
            return [1,0,0,0]
        elif x < thresholds[1]:
            return [0,1,0,0]
        else:
            return [0,0,1,0]
    # end to_category

    def compute_harmonic_rhythm_density(self, chord_token_ids):
        """
        Compute harmonic rhythm density: average number of chord changes
        per bar that contains at least one valid chord.
        
        Args:
            chord_token_ids (list[int]): sequence of tokens including bar, chord, nc, and pad.
            
        Returns:
            float: average number of chord changes per bar with chords,
                or 0.0 if no valid bar exists.
        """
        bars = []
        current_bar = []

        # Split into bars
        for tok in chord_token_ids:
            if tok == self.bar_token_id:
                if current_bar:  # save previous bar
                    bars.append(current_bar)
                current_bar = []
            else:
                current_bar.append(tok)
        if current_bar:  # last bar
            bars.append(current_bar)

        chord_counts = []

        for bar in bars:
            valid_chords = [t for t in bar if t not in (self.nc_token_id, self.pad_token_id)]
            if not valid_chords:
                continue  # skip bars without valid chords

            # Count chord changes (successive unique chord IDs)
            changes = 1  # at least one chord if valid_chords is not empty
            for prev, curr in zip(valid_chords, valid_chords[1:]):
                if curr != prev:
                    changes += 1

            chord_counts.append(changes)

        if not chord_counts:
            return 0.0  # no valid bars
        
        hrd = sum(chord_counts) / len(chord_counts)
        return hrd, self.to_category(hrd, [1.0001, 1.5556])
    # end compute_harmonic_rhythm_density

    def compute_harmonic_complexity(self, chord_tokens):
        """
        Compute harmonic complexity as the information entropy of the average
        pitch class profile across the given chord sequence.

        Parameters
        ----------
        chord_tokens : list
            List of chord tokens (strings or ids, depending on how pitch_class_from_chord_token works).

        Returns
        -------
        float
            Harmonic complexity (entropy in nats).
        """
        if not chord_tokens:
            return 0.0

        # Accumulate pitch class profiles
        pitch_class_sum = np.zeros(12, dtype=float)
        for chord_token in chord_tokens:
            pc_profile = self.pitch_class_from_chord_token(chord_token)  # 12D binary vector
            pitch_class_sum += pc_profile

        # Normalize to a probability distribution
        if np.sum(pitch_class_sum) == 0:
            return 0.0
        pitch_class_dist = pitch_class_sum / np.sum(pitch_class_sum)

        # Compute entropy (natural log)
        entropy = -np.sum(pitch_class_dist * np.log(pitch_class_dist + 1e-12))

        return float(entropy), self.to_category(entropy, [1.8225, 1.9254])
    # end compute_harmonic_complexity

    def encode(
            self,
            file_path,
            filler_token='<nc>',
            keep_durations=False,
            normalize_tonality=False
        ):
        file_ext = file_path.split('.')[-1]
        
        if file_ext in ['xml', 'mxl', 'musicxml']:
            return self.encode_musicXML(
                file_path,
                filler_token=filler_token,
                keep_durations=keep_durations,
                normalize_tonality=normalize_tonality
            )
        elif file_ext in ['mid', 'midi']:
            return self.encode_MIDI(
                file_path,
                filler_token=filler_token,
                keep_durations=keep_durations,
                normalize_tonality=normalize_tonality
            )
        else:
            print('ERROR: unknown file extension:', file_ext)
    # end encode

    def encode_musicXML(
            self,
            file_path,
            filler_token='<nc>',
            keep_durations=False,
            normalize_tonality=False
        ):
        # Load the score and flatten
        score = converter.parse(file_path)
        if normalize_tonality:
            # Detect original key
            original_key = detect_key(score)
            # Transpose to C major or A minor
            to_c_or_a_interval = get_transposition_interval(original_key)
            score = transpose_score(score, to_c_or_a_interval)
            # Keep interval to transpose back to original key later
            back_interval = to_c_or_a_interval.reverse()

        time_signature = score.recurse().getElementsByClass(meter.TimeSignature).first()
        ts_num_list = [0]*14
        ts_den_list = [0,0]
        ts_num_list[ int( min( max(time_signature.numerator-2,0) , 13) ) ] = 1
        ts_den_list[ int( time_signature.denominator == 4 ) ] = 1
        melody_part = score.parts[0].flatten()
        chords_part = None
        if len(score.parts) > 1:
            chords_part = score.parts[1].chordify().flatten()
        # Define quantization note length
        if self.quantization == '16th':
            ql_per_quantum = 1 / 4
        elif self.quantization == '8th':
            ql_per_quantum = 1 / 2
        elif self.quantization == '4th':
            ql_per_quantum = 1
        elif self.quantization == '32nd':
            ql_per_quantum = 1 / 8
        else: # assume 16th
            ql_per_quantum = 1 / 4

        # Find first chord symbol and bar to trim before it
        first_chord_offset = None
        skip_steps = 0
        if self.trim_start:
            if chords_part is None:
                for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
                    first_chord_offset = el.offset
                    break
            else:
                for el in chords_part:
                    first_chord_offset = el.offset
                    break
            measure_start_offset = 0.0
            if first_chord_offset is not None:
                for meas in melody_part.getElementsByClass(stream.Measure):
                    if meas.offset <= first_chord_offset < meas.offset + meas.duration.quarterLength:
                        measure_start_offset = meas.offset
                        break
            skip_steps = int(np.round(measure_start_offset / ql_per_quantum))

        # Determine total length in quantum length notes
        total_duration_q = melody_part.highestTime
        total_steps = int(np.ceil(total_duration_q / ql_per_quantum))
        
        # initialize chord tokens
        chord_tokens = [None] * total_steps
        chord_token_ids = [self.pad_token_id] * total_steps
        
        # Fill chord grid
        if chords_part is None:
            for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
                start = int(np.floor(el.offset / ql_per_quantum))
                if 0 <= start < len(chord_tokens):
                    chord_tokens[start], chord_token_ids[start] = self.handle_chord_symbol(el)
                if keep_durations:
                    end = int(np.ceil( (el.offset + el.duration.quarterLength) / ql_per_quantum)) + 1
                    if end < len(chord_tokens):
                        chord_tokens[end] = '<nc>'
                        chord_token_ids[end] = self.vocab['<nc>']
        else:
            for el in chords_part.recurse().getElementsByClass(chord.Chord):
                start = int(np.round(el.offset / ql_per_quantum))
                if 0 <= start < len(chord_tokens):
                    chord_tokens[start], chord_token_ids[start] = self.handle_chord_symbol(el)
                if keep_durations:
                    end = int(np.round( (el.offset + el.duration.quarterLength) / ql_per_quantum)) + 1
                    if end < len(chord_tokens):
                        chord_tokens[end] = '<nc>'
                        chord_token_ids[end] = self.vocab['<nc>']

        # Propagate chord forward
        for i in range(1, len(chord_tokens)):
            if chord_tokens[i] is None:
                chord_tokens[i] = chord_tokens[i-1]
                chord_token_ids[i] = chord_token_ids[i-1]

        # Fill missing with <pad> or <nc>
        for i in range(len(chord_tokens)):
            if chord_tokens[i] is None:
                chord_tokens[i] = filler_token
                chord_token_ids[i] = self.vocab[filler_token]

        # compute full range pianoroll
        # Get pitch range: MIDI 21 (A0) to 108 (C8) -> 88 notes
        pitch_range = list(range(21, 109))
        n_pitch = len(pitch_range)
        # Allocate raw matrices (we will trim/pad later)
        raw_pianoroll = np.zeros((total_steps, n_pitch), dtype=np.uint8)

        # Fill pianoroll
        for el in melody_part.notesAndRests:
            start = int(np.floor(el.offset / ql_per_quantum))
            dur_steps = int(np.ceil(el.quarterLength / ql_per_quantum))
            
            if isinstance(el, note.Note):
                midi = el.pitch.midi
                if midi in pitch_range:
                    idx = pitch_range.index(midi)
                    raw_pianoroll[start:start+dur_steps, idx] = 1

            elif isinstance(el, chord.Chord):  # Just in case
                for pitch in el.pitches:
                    midi = pitch.midi
                    if midi in pitch_range:
                        idx = pitch_range.index(midi)
                        raw_pianoroll[start:start+dur_steps, idx] = 1

        # compute pitch class profile (top 12 dims) if needed
        n_steps = raw_pianoroll.shape[0]
        if self.use_pc_roll:
            pitch_classes = np.zeros((n_steps, 12), dtype=np.uint8)
            for i in range(n_steps):
                pitch_indices = np.where(raw_pianoroll[i] > 0)[0]
                for idx in pitch_indices:
                    midi = pitch_range[idx]
                    pitch_classes[i, midi % 12] = 1
            if self.use_full_range_melody:
                full_pianoroll = np.hstack([pitch_classes, raw_pianoroll])  # Shape: (T, 12 + 88)
            else:
                full_pianoroll = pitch_classes  # Shape: (T, 12)
        else:
            full_pianoroll = raw_pianoroll  # Shape: (T, 88)
        
        # intertwine bar information in pianoroll and harmony tokens
        if self.intertwine_bar_info:
            # --- Prepare ---
            num_steps = full_pianoroll.shape[0]  # current time steps (rows)
            num_pitches = full_pianoroll.shape[1]  # columns = pitches/features
            bar_column = np.zeros((num_steps, 1), dtype=np.float32)  # all-zero bar column
            full_pianoroll = np.hstack([full_pianoroll, bar_column])  # add extra column at the end

            # Compute insertion indices (where bars start)
            # Assuming you already know bar_step positions from melody or measure offsets
            insertion_indices = []
            for meas in score.parts[0].getElementsByClass(stream.Measure):
                bar_step = int(np.round(meas.offset / ql_per_quantum))
                if 0 <= bar_step <= full_pianoroll.shape[0]:  # allow insertion at end too
                    insertion_indices.append(bar_step)

            # Make sure sorted and unique
            insertion_indices = sorted(set(insertion_indices))

            # --- Insert bar columns into pianoroll ---
            pianoroll_ext = []
            step = 0
            for i in range(full_pianoroll.shape[0]):
                if i in insertion_indices:
                    # insert a bar row
                    bar_row = np.zeros((1, full_pianoroll.shape[1]), dtype=np.float32)
                    bar_row[0, -1] = 1.0  # mark in the bar column
                    pianoroll_ext.append(bar_row)
                # always add the original step (aligned)
                if step < full_pianoroll.shape[0]:
                    pianoroll_ext.append(full_pianoroll[step:(step + 1), :])
                    step += 1
            
            full_pianoroll = np.vstack(pianoroll_ext)

            # --- Insert bar tokens into chord sequence ---
            chord_tokens_ext = []
            chord_token_ids_ext = []
            step = 0
            for i in range(len(chord_tokens)):
                if i in insertion_indices:  # bar insertion
                    chord_tokens_ext.append(self.bar_token)
                    chord_token_ids_ext.append(self.bar_token_id)
                if step < len(chord_tokens):
                    chord_tokens_ext.append(chord_tokens[step])
                    chord_token_ids_ext.append(chord_token_ids[step])
                    step += 1

            chord_tokens = chord_tokens_ext
            chord_token_ids = chord_token_ids_ext
        # end intertwine bars

        # Trim to start at first chord bar
        if self.trim_start:
            raw_pianoroll = raw_pianoroll[skip_steps:]
            chord_tokens = chord_tokens[skip_steps:]
            chord_token_ids = chord_token_ids[skip_steps:]
        
        # Apply fixed length (pad or trim)
        n_steps = len(chord_tokens) # should theoretically be equal to len(full_pianoroll)
        if self.fixed_length is not None:
            if n_steps >= self.fixed_length:
                full_pianoroll = full_pianoroll[:self.fixed_length]
                chord_tokens = chord_tokens[:self.fixed_length]
                chord_token_ids = chord_token_ids[:self.fixed_length]
                attention_mask = [1]*self.fixed_length
            elif n_steps < self.fixed_length:
                pad_len = self.fixed_length - n_steps
                pad_pr = np.zeros((pad_len, full_pianoroll.shape[1]), dtype=np.uint8)
                pad_ch = ["<pad>"] * pad_len
                pad_ch_ids = [self.vocab["<pad>"]] * pad_len
                full_pianoroll = np.vstack([full_pianoroll, pad_pr])
                chord_tokens += pad_ch
                chord_token_ids += pad_ch_ids
                attention_mask = [1]*n_steps + [0]*pad_len
        else:
            attention_mask = [1]*n_steps

        # compute rhythm density and harmonic complexity
        # each function returns a numerical value
        # that will later be translated into a
        # 4D binary array with categories:
        # [1,0,0,0]: low
        # [0,1,0,0]: medium
        # [0,0,1,0]: high
        # [0,0,0,1]: no condition
        h_rhythm, r_cat = self.compute_harmonic_rhythm_density(chord_token_ids)
        h_complexity, c_cat = self.compute_harmonic_complexity(chord_tokens)

        return {
            'harmony_tokens': chord_tokens,
            'harmony_ids': chord_token_ids,
            'pianoroll': full_pianoroll,
            'time_signature': ts_num_list + ts_den_list,
            'attention_mask': attention_mask,
            'skip_steps': skip_steps,
            'melody_part': melody_part,
            'ql_per_quantum': ql_per_quantum,
            'back_interval': back_interval if normalize_tonality else None,
            'harmonic_rhythm_density': h_rhythm,
            'harmonic_complexity': h_complexity,
            'h_density_complexity': r_cat + c_cat
        }
    # end encode_musicXML

    def encode_MIDI(
            self,
            file_path,
            filler_token='<nc>',
            keep_durations=False,
            normalize_tonality=False
        ):
        # Load the score and flatten
        score = converter.parse(file_path)
        if normalize_tonality:
            # Detect original key
            original_key = detect_key(score)
            # Transpose to C major or A minor
            to_c_or_a_interval = get_transposition_interval(original_key)
            score = transpose_score(score, to_c_or_a_interval)
            # Keep interval to transpose back to original key later
            back_interval = to_c_or_a_interval.reverse()
        
        time_signature = score.recurse().getElementsByClass(meter.TimeSignature).first()
        ts_num_list = [0]*14
        ts_den_list = [0,0]
        ts_num_list[ int( min( max(time_signature.numerator-2,0) , 13) ) ] = 1
        ts_den_list[ int( time_signature.denominator == 4 ) ] = 1
        melody_part = score.parts[0].flatten()
        chords_part = None
        if len(score.parts) > 1:
            chords_part = score.parts[1].chordify().flatten()
        # Define quantization note length
        if self.quantization == '16th':
            ql_per_quantum = 1 / 4
        elif self.quantization == '8th':
            ql_per_quantum = 1 / 2
        elif self.quantization == '4th':
            ql_per_quantum = 1
        elif self.quantization == '32nd':
            ql_per_quantum = 1 / 8
        else: # assume 16th
            ql_per_quantum = 1 / 4

        # Find first chord symbol and bar to trim before it
        first_chord_offset = None
        skip_steps = 0
        if self.trim_start:
            if chords_part is not None:
                for el in chords_part.recurse().getElementsByClass(chord.Chord):
                    first_chord_offset = el.offset
                    break
            measure_start_offset = 0.0
            if first_chord_offset is not None:
                for meas in melody_part.getElementsByClass(stream.Measure):
                    if meas.offset <= first_chord_offset < meas.offset + meas.duration.quarterLength:
                        measure_start_offset = meas.offset
                        break
            skip_steps = int(np.round(measure_start_offset / ql_per_quantum))

        # Determine total length in quantum length notes
        total_duration_q = melody_part.highestTime
        total_steps = int(np.ceil(total_duration_q / ql_per_quantum))

        # initialize chord tokens
        chord_tokens = [None] * total_steps
        chord_token_ids = [self.pad_token_id] * total_steps

        # Fill chord grid
        if chords_part is not None:
            for el in chords_part.recurse().getElementsByClass(chord.Chord):
                start = int(np.floor(el.offset / ql_per_quantum))
                if 0 <= start < len(chord_tokens):
                    chord_tokens[start], chord_token_ids[start] = self.handle_chord_symbol(el)
                if keep_durations:
                    end = int(np.ceil( (el.offset + el.duration.quarterLength) / ql_per_quantum) )# + 1
                    if end < len(chord_tokens) and chord_tokens[end] is None:
                        chord_tokens[end] = '<nc>'
                        chord_token_ids[end] = self.vocab['<nc>']
            
            # Propagate chord forward
            for i in range(1, len(chord_tokens)):
                if chord_tokens[i] is None:
                    chord_tokens[i] = chord_tokens[i-1]
                    chord_token_ids[i] = chord_token_ids[i-1]
            
            # Fill missing with <pad> or <nc>
            for i in range(len(chord_tokens)):
                if chord_tokens[i] is None:
                    chord_tokens[i] = filler_token
                    chord_token_ids[i] = self.vocab[filler_token]
        
        # compute full range pianoroll
        # Get pitch range: MIDI 21 (A0) to 108 (C8) -> 88 notes
        pitch_range = list(range(21, 109))
        n_pitch = len(pitch_range)
        # Allocate raw matrices (we will trim/pad later)
        raw_pianoroll = np.zeros((total_steps, n_pitch), dtype=np.uint8)

        # Fill pianoroll
        for el in melody_part.notesAndRests:
            start = int(np.floor(el.offset / ql_per_quantum))
            dur_steps = int(np.ceil(el.quarterLength / ql_per_quantum))
            
            if isinstance(el, note.Note):
                midi = el.pitch.midi
                if midi in pitch_range:
                    idx = pitch_range.index(midi)
                    raw_pianoroll[start:start+dur_steps, idx] = 1

            elif isinstance(el, chord.Chord):  # Just in case
                for pitch in el.pitches:
                    midi = pitch.midi
                    if midi in pitch_range:
                        idx = pitch_range.index(midi)
                        raw_pianoroll[start:start+dur_steps, idx] = 1

        # compute pitch class profile (top 12 dims) if needed
        n_steps = len(raw_pianoroll)
        if self.use_pc_roll:
            pitch_classes = np.zeros((n_steps, 12), dtype=np.uint8)
            for i in range(n_steps):
                pitch_indices = np.where(raw_pianoroll[i] > 0)[0]
                for idx in pitch_indices:
                    midi = pitch_range[idx]
                    pitch_classes[i, midi % 12] = 1
            if self.use_full_range_melody:
                full_pianoroll = np.hstack([pitch_classes, raw_pianoroll])  # Shape: (T, 12 + 88)
            else:
                full_pianoroll = pitch_classes  # Shape: (T, 12)
        else:
            full_pianoroll = raw_pianoroll  # Shape: (T, 88)
        
        # intertwine bar information in pianoroll and harmony tokens
        if self.intertwine_bar_info:
            # --- Prepare ---
            num_steps = full_pianoroll.shape[0]  # current time steps (rows)
            num_pitches = full_pianoroll.shape[1]  # columns = pitches/features
            bar_column = np.zeros((num_steps, 1), dtype=np.float32)  # all-zero bar column
            full_pianoroll = np.hstack([full_pianoroll, bar_column])  # add extra column at the end

            # Compute insertion indices (where bars start)
            # Assuming you already know bar_step positions from melody or measure offsets
            insertion_indices = []
            for meas in score.parts[0].getElementsByClass(stream.Measure):
                bar_step = int(np.round(meas.offset / ql_per_quantum))
                if 0 <= bar_step <= full_pianoroll.shape[0]:  # allow insertion at end too
                    insertion_indices.append(bar_step)

            # Make sure sorted and unique
            insertion_indices = sorted(set(insertion_indices))

            # --- Insert bar columns into pianoroll ---
            pianoroll_ext = []
            step = 0
            for i in range(full_pianoroll.shape[0]):
                if i in insertion_indices:
                    # insert a bar row
                    bar_row = np.zeros((1, full_pianoroll.shape[1]), dtype=np.float32)
                    bar_row[0, -1] = 1.0  # mark in the bar column
                    pianoroll_ext.append(bar_row)
                # always add the original step (aligned)
                if step < full_pianoroll.shape[0]:
                    pianoroll_ext.append(full_pianoroll[step:(step + 1), :])
                    step += 1
            
            full_pianoroll = np.vstack(pianoroll_ext)

            # --- Insert bar tokens into chord sequence ---
            chord_tokens_ext = []
            chord_token_ids_ext = []
            step = 0
            for i in range(len(chord_tokens)):
                if i in insertion_indices:  # bar insertion
                    chord_tokens_ext.append(self.bar_token)
                    chord_token_ids_ext.append(self.bar_token_id)
                if step < len(chord_tokens):
                    chord_tokens_ext.append(chord_tokens[step])
                    chord_token_ids_ext.append(chord_token_ids[step])
                    step += 1

            chord_tokens = chord_tokens_ext
            chord_token_ids = chord_token_ids_ext
        # end intertwine bars

        # Trim to start at first chord bar
        if self.trim_start:
            raw_pianoroll = raw_pianoroll[skip_steps:]
            chord_tokens = chord_tokens[skip_steps:]
            chord_token_ids = chord_token_ids[skip_steps:]
            
        # Apply fixed length (pad or trim)
        n_steps = len(chord_tokens) # should theoretically be equal to len(full_pianoroll)
        if self.fixed_length is not None:
            if n_steps >= self.fixed_length:
                full_pianoroll = full_pianoroll[:self.fixed_length]
                chord_tokens = chord_tokens[:self.fixed_length]
                chord_token_ids = chord_token_ids[:self.fixed_length]
                attention_mask = [1]*self.fixed_length
            elif n_steps < self.fixed_length:
                pad_len = self.fixed_length - n_steps
                pad_pr = np.zeros((pad_len, full_pianoroll.shape[1]), dtype=np.uint8)
                pad_ch = ["<pad>"] * pad_len
                pad_ch_ids = [self.vocab["<pad>"]] * pad_len
                full_pianoroll = np.vstack([full_pianoroll, pad_pr])
                chord_tokens += pad_ch
                chord_token_ids += pad_ch_ids
                attention_mask = [1]*n_steps + [0]*pad_len
        else:
            attention_mask = [1]*n_steps

        # compute rhythm density and harmonic complexity
        # each function returns a numerical value
        # that will later be translated into a
        # 4D binary array with categories:
        # [1,0,0,0]: low
        # [0,1,0,0]: medium
        # [0,0,1,0]: high
        # [0,0,0,1]: no condition
        h_rhythm, r_cat = self.compute_harmonic_rhythm_density(chord_token_ids)
        h_complexity, c_cat = self.compute_harmonic_complexity(chord_tokens)

        return {
            'harmony_tokens': chord_tokens,
            'harmony_ids': chord_token_ids,
            'pianoroll': full_pianoroll,
            'time_signature': ts_num_list + ts_den_list,
            'attention_mask': attention_mask,
            'skip_steps': skip_steps,
            'melody_part':melody_part,
            'ql_per_quantum': ql_per_quantum,
            'back_interval': back_interval if normalize_tonality else None,
            'harmonic_rhythm_density': h_rhythm,
            'harmonic_complexity': h_complexity,
            'h_density_complexity': r_cat + c_cat
        }
    # end encode_MIDI

    def fit_transform(self, corpus, add_start_harmony_token=True):
        self.fit(corpus)
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end transform

    def __call__(self, corpus, add_start_harmony_token=True):
        return self.transform(corpus, add_start_harmony_token=add_start_harmony_token)
    # end __call__

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save special tokens and configuration
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        config = {"special_tokens": self.special_tokens}
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    # end save_pretrained

    @classmethod
    def from_pretrained(cls, load_directory):
        # Load vocabulary
        vocab_file = os.path.join(load_directory, "vocab.json")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Load special tokens and configuration
        config_file = os.path.join(load_directory, "tokenizer_config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        special_tokens = config.get("special_tokens", {})
        
        # Create a new tokenizer instance
        return cls(vocab, special_tokens)
    # end from_pretrained

# end class CSGridMLMTokenizer