import torch
import torch.nn.functional as F
from music21 import harmony, stream, metadata, chord, note, key, meter, tempo, duration
from model import SEVanilla
import mir_eval
import numpy as np
from copy import deepcopy
import os
from music_utils import transpose_score

def remove_conflicting_rests(flat_part):
    """
    Remove any Rest in a flattened part whose offset coincides with a Note.
    Assumes the input stream is already flattened.
    This also tries to fix broken duration values that have come as a result of
    flattening.
    """
    cleaned = stream.Part()
    all_notes = [el for el in flat_part if isinstance(el, note.Note)]
    note_offsets = [n.offset for n in all_notes]
    note_durations = [n.offset for n in all_notes]
    for i, n in enumerate(all_notes):
        if n.duration.quarterLength == 0:
            if i < len(all_notes)-1:
                n.duration = duration.Duration( note_offsets[i+1] - note_offsets[i] )
            else:
                n.duration = duration.Duration( 0.5 )
        # if i < len(all_notes)-1:
        #     if note_durations[i] > note_offsets[i+1] - note_offsets[i]:
        #         n.duration = duration.Duration( note_offsets[i+1] - note_offsets[i] )

    for el in flat_part:
        # Skip Rest if it shares offset with a Note
        if isinstance(el, note.Rest) and el.offset in note_offsets:
            continue
        cleaned.insert(el.offset, el)

    return cleaned
# end remove_conflicting_rests

def overlay_generated_harmony(melody_part, generated_chords, ql_per_16th, skip_steps):
    # create a part for chords in midi format
    # melody_part = melody_part.makeMeasures()
    # chords_part = deepcopy(melody_part)
    # Create deep copy of flat melody part
    # Create a new part for filtered content
    filtered_part = stream.Part()
    # filtered_part.id = melody_part.id  # Preserve ID

    # Copy key and time signatures from the original part
    for el in melody_part.recurse().getElementsByClass((key.KeySignature, meter.TimeSignature,  tempo.MetronomeMark)):
        if el.offset < 64:
            filtered_part.insert(el.offset, el)

    # Copy notes and rests with offset < 64
    for el in melody_part.flatten().notesAndRests:
        if el.offset < 64:
            filtered_part.insert(el.offset, el)
    # clear conflicting rests with notes of the same offset
    filtered_part = remove_conflicting_rests(filtered_part)

    # Replace the original part with the filtered one
    melody_part = filtered_part
    
    # Remove old chord symbols
    for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
        melody_part.remove(el)
    
    # Prepare for clamping durations — convert melody to measures
    melody_measures = melody_part.makeMeasures()

    chords_part = deepcopy(melody_measures)
    # Strip musical elements but retain structure
    for measure in chords_part.getElementsByClass(stream.Measure):
        for el in list(measure):
            if isinstance(el, (note.Note, note.Rest, chord.Chord, harmony.ChordSymbol)):
                measure.remove(el)
        # Add a placeholder full-measure rest to preserve the measure
        full_rest = note.Rest()
        full_rest.quarterLength = measure.barDuration.quarterLength
        measure.insert(0.0, full_rest)

    # Track inserted chords
    last_chord_symbol = None
    inserted_chords = {}

    # keep bar tokens out of the steps count
    num_bar_tokens = 0
    for i, mir_chord in enumerate(generated_chords):
        if mir_chord in ("<pad>", "<nc>"):
            continue
        if mir_chord == last_chord_symbol:
            continue
        if mir_chord == "<bar>":
            num_bar_tokens += 1
            continue
        
        offset = (i + skip_steps - num_bar_tokens) * ql_per_16th
        
        # Decode mir_eval chord symbol to chord symbol object
        try:
            r, t, _ = mir_eval.chord.encode(mir_chord, reduce_extended_chords=True)
            pcs = r + np.where(t > 0)[0] + 48
            c = chord.Chord(pcs.tolist())
            chord_symbol = harmony.chordSymbolFromChord(c)
        except Exception as e:
            print(f"Skipping invalid chord {mir_chord} at step {i}: {e}")
            continue
        
        # Clamp duration so it doesn't overflow into next bar
        bars = list(chords_part.getElementsByClass(stream.Measure))
        for b in reversed(bars):
            if b.offset <= offset:
                bar = b
                break
        # bar = next((b for b in reversed(bars) if b.offset <= offset), None)

        offset = (i + skip_steps - num_bar_tokens) * ql_per_16th
        
        if bar:
            bar_start = bar.offset
            bar_end = bar_start + bar.barDuration.quarterLength
            max_duration = bar_end - offset
            c.quarterLength = min(c.quarterLength, max_duration)
            # chord_symbol.quarterLength = min(c.quarterLength, max_duration)
        # chords_part.insert(offset, c)
        # Remove any placeholder rests at 0.0
        for el in bar.getElementsByOffset(0.0):
            if isinstance(el, note.Rest):
                bar.remove(el)
        bar.insert(offset - bar_start, c)
        # harmonized_part.insert(offset, chord_symbol)
        inserted_chords[i] = chord_symbol
        last_chord_symbol = mir_chord

    # Repeat previous chord at start of bars with no chord
    for m in chords_part.getElementsByClass(stream.Measure):
        bar_offset = m.offset
        bar_duration = m.barDuration.quarterLength
        # has_chord = any(isinstance(el, chord.Chord) and el.offset == bar_offset for el in m)
        # has_chord = any( isinstance(el, chord.Chord) for el in m )
        has_chord = any(isinstance(el, chord.Chord) and el.offset == 0. for el in m)
        if not has_chord:
            # Find previous chord before this measure
            prev_chords = []
            for curr_bar in chords_part.recurse().getElementsByClass(stream.Measure):
                for el in curr_bar.recurse().getElementsByClass(chord.Chord):
                        if curr_bar.offset + el.offset < bar_offset:
                            prev_chords.append(el)
            if prev_chords:
                # Remove any placeholder rests at 0.0
                for el in m.getElementsByOffset(0.0):
                    if isinstance(el, note.Rest):
                        m.remove(el)
                prev_chord = prev_chords[-1]
                m.insert(0.0, deepcopy(prev_chord))
        else:
            # Remove any placeholder rests at 0.0
            for el in m.getElementsByOffset(0.0):
                if isinstance(el, note.Rest):
                    m.remove(el)
            # modify duration so that it doesn't affect the next bar
            for el in m.notes:
                if isinstance(el, chord.Chord):
                    max_duration = bar_duration - el.offset
                    if el.quarterLength > max_duration:
                        el.quarterLength = max_duration

    # Create final score with chords and melody
    score = stream.Score()
    score.insert(0, melody_measures)
    score.insert(0, chords_part)

    return score
# end overlay_generated_harmony

def save_harmonized_score(score, title="Harmonized Piece", out_path="harmonized.xml"):
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    if out_path.endswith('.xml') or out_path.endswith('.mxl') or out_path.endswith('.musicxml'):
        score.write('musicxml', fp=out_path)
    elif out_path.endswith('.mid') or out_path.endswith('.midi'):
        score.write('midi', fp=out_path)
    else:
        print('uknown file format for file: ', out_path)
# end save_harmonized_score

def load_SEVanilla(
        tokenizer,
        device,
        d_model=512,
        checkpoint_path=None,
    ):
    transformer_model = SEVanilla(
        chord_vocab_size=len(tokenizer.vocab),
        device=device,
        d_model=d_model,
        nhead=8,
        num_layers=8,
        grid_length=80,
        pianoroll_dim=tokenizer.pianoroll_dim,
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(f'saved_models/f2f2f_pretrained/pretrained.pt', map_location=device)
        # checkpoint = torch.load(f'saved_models/SELaC_pretrained/pretrained.pt', map_location=device)
    transformer_model.load_state_dict(checkpoint)
    transformer_model.to(device)
    transformer_model.eval()
    return transformer_model
# end load_FiLMSEModel

def nucleus_token_by_token_generate(
        model,
        melody_grid,            # (1, seq_len, input_dim)
        mask_token_id,        # token ID used for masking
        temperature=1.0,        # optional softmax temperature
        pad_token_id=None,      # token ID for <pad>
        nc_token_id=None,       # token ID for <nc>
        force_fill=True,        # disallow <pad>/<nc> before melody ends
        chord_constraints=None, # chord + bar constraints
        p=0.9,                  # nucleus threshold
        unmasking_order='certain', # in ['random', 'start', 'end', 'certain', 'uncertain']
    ):
    device = melody_grid.device
    seq_len = melody_grid.shape[1]

    # --- 1. Initialize ---
    visible_harmony = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    if chord_constraints is not None:
        idxs = torch.logical_and(chord_constraints != nc_token_id,
                                 chord_constraints != pad_token_id)
        visible_harmony[idxs] = chord_constraints[idxs]
    # Compute last active melody index if forcing fill
    if force_fill:
        active = (melody_grid != 0).any(dim=-1).squeeze(0)  # shape: (seq_len,)
        try:
            last_active_index = active.nonzero(as_tuple=True)[0].max().item()
        except:
            last_active_index = -1
    else:
        last_active_index = -1

    step = 0
    
    while (visible_harmony == mask_token_id).any():
        with torch.no_grad():
            logits = model(
                melody_grid=melody_grid.to(model.device),
                harmony_tokens=visible_harmony.to(model.device),
            )  # (1, seq_len, vocab_size)
        # --- Masked position selection ---
        masked_positions = (visible_harmony == mask_token_id).squeeze(0).nonzero(as_tuple=True)[0]
        if masked_positions.numel() == 0:
            break
        masked_positions = masked_positions.to(model.device)
        probs = torch.softmax(logits[0, masked_positions] / temperature, dim=-1)
        entropies = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)

        if unmasking_order == 'random':
            pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()
        elif unmasking_order == 'uncertain':
            pos = masked_positions[torch.argmax(entropies)].item()
        elif unmasking_order == 'certain':
            pos = masked_positions[torch.argmin(entropies)].item()
        elif unmasking_order == 'start':
            pos = masked_positions[0].item()
        elif unmasking_order == 'end':
            pos = masked_positions[-1].item()
        else:
            pos = masked_positions[torch.randint(0, masked_positions.numel(), (1,))].item()

        # Mask out invalid predictions if enforcing force_fill
        if force_fill and (pad_token_id is not None and nc_token_id is not None):
            for i in range(seq_len):
                if i <= last_active_index:
                    logits[0, i, pad_token_id] = float('-inf')
                    logits[0, i, nc_token_id] = float('-inf')
                else:
                    logits[0, i, :] = float('-inf')
                    logits[0, i, pad_token_id] = 1.0

        # --- Nucleus sampling step ---
        logits_pos = logits[0, pos] / temperature
        logits_pos[ mask_token_id ] = logits_pos.min().item()/100  # prevent selecting mask token
        probs_pos = torch.softmax(logits_pos, dim=-1)

        # sort probs descending
        sorted_probs, sorted_idx = torch.sort(probs_pos, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mask out tokens beyond nucleus p
        nucleus_mask = cumulative_probs <= p
        nucleus_mask[0] = True  # keep at least one token
        nucleus_probs = sorted_probs[nucleus_mask]
        nucleus_idx = sorted_idx[nucleus_mask]

        # renormalize
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        # sample
        sampled_idx = torch.multinomial(nucleus_probs, 1).item()
        token = nucleus_idx[sampled_idx].item()

        # update harmony
        visible_harmony[0, pos] = token
        step += 1
    
    return visible_harmony
# end nucleus_token_by_token_generate

def generate_files_with_nucleus(
        model,
        tokenizer,
        input_f_path,
        mxl_folder_out,
        midi_folder_out,
        name_suffix,
        use_constraints=False,
        intertwine_bar_info=True,
        normalize_tonality=True,
        temperature=1.0,
        p=0.9,
        unmasking_order='certain',
        create_gen=True,
        create_real=False
    ):
    # we cannot have intertwine_bar_info == True and use_constraints == False
    # because bar information is passed through the constraints
    # if intertwine_bar_info:
    #     use_constraints = True

    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id

    input_encoded = tokenizer.encode(
        input_f_path,
        keep_durations=True,
        normalize_tonality=normalize_tonality,
    )

    harmony_real = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
    harmony_input = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
    # if intertwine_bar_info is True and use_constraints is False, we only need to pass
    # the bar information as a constraint, not the chords, or anything else
    # so mask out everything except from bar_token_ids
    if intertwine_bar_info and not use_constraints:
        harmony_input[ harmony_input != tokenizer.bar_token_id ] = tokenizer.mask_token_id
    melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
    
    if create_gen:
        gen_harmony = nucleus_token_by_token_generate(
            model=model,
            melody_grid=melody_grid.to(model.device),
            mask_token_id=tokenizer.mask_token_id,
            temperature=temperature,
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True,         # disallow <pad>/<nc> before melody ends
            chord_constraints = harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
            p=p,
            unmasking_order=unmasking_order
        )
        nucleus_generated_harmony = gen_harmony
        gen_output_tokens = []
        for t in nucleus_generated_harmony[0].tolist():
            gen_output_tokens.append( tokenizer.ids_to_tokens[t] )
    else:
        gen_output_tokens = None
        nucleus_generated_harmony = None
    # keep ground truth
    harmony_real_tokens = []
    for t in harmony_real[0].tolist():
        harmony_real_tokens.append( tokenizer.ids_to_tokens[t] )
    gen_score = None
    real_score = None
    if create_gen:
        gen_score = overlay_generated_harmony(
            input_encoded['melody_part'],
            gen_output_tokens,
            input_encoded['ql_per_quantum'],
            input_encoded['skip_steps']
        )
        if normalize_tonality:
            gen_score = transpose_score(gen_score, input_encoded['back_interval'])
        if mxl_folder_out is not None:
            os.makedirs(mxl_folder_out, exist_ok=True)
            mxl_file_name = os.path.join(mxl_folder_out, f'gen_{name_suffix}' + '.mxl')
            save_harmonized_score(gen_score, out_path=mxl_file_name)
        if midi_folder_out is not None:
            os.makedirs(midi_folder_out, exist_ok=True)
            midi_file_name = os.path.join(midi_folder_out, f'gen_{name_suffix}' + '.mid')
            save_harmonized_score(gen_score, out_path=midi_file_name)
        # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')
    if create_real:
        real_score = overlay_generated_harmony(
            input_encoded['melody_part'],
            harmony_real_tokens,
            input_encoded['ql_per_quantum'],
            input_encoded['skip_steps']
        )
        
        if normalize_tonality:
            real_score = transpose_score(real_score, input_encoded['back_interval'])
        if mxl_folder_out is not None:
            os.makedirs(mxl_folder_out, exist_ok=True)
            mxl_file_name = os.path.join(mxl_folder_out, f'real_{name_suffix}' + '.mxl')
            save_harmonized_score(real_score, out_path=mxl_file_name)
        if midi_folder_out is not None:
            os.makedirs(midi_folder_out, exist_ok=True)
            midi_file_name = os.path.join(midi_folder_out, f'real_{name_suffix}' + '.mid')
            save_harmonized_score(real_score, out_path=midi_file_name)
        # os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')

    return {
        'gen_output_tokens': gen_output_tokens,
        'gen_output_token_ids': nucleus_generated_harmony,
        'harmony_real_tokens': harmony_real_tokens,
        'gen_score': gen_score,
        'real_score': real_score,
    }
# end generate_files_with_nucleus
