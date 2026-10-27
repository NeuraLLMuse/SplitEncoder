from music21 import converter, key, pitch, interval, midi

def detect_key(score):
    """Detect the most likely key of the piece."""
    return score.analyze('key')
# end detect_key

def get_transposition_interval(k):
    """Get interval needed to transpose to C major or A minor."""
    if k.mode == 'major':
        return interval.Interval(k.tonic, pitch.Pitch('C'))
    elif k.mode == 'minor':
        return interval.Interval(k.tonic, pitch.Pitch('A'))
    else:
        # Default to no transposition
        return interval.Interval(0)
# end get_transposition_interval

def transpose_score(score, transposition_interval):
    """Transpose the score by the given interval."""
    return score.transpose(transposition_interval)
# end transpose_score

# def process_music(input_file_path, output_file_path, process_function):
#     """Load, normalize key, process, and transpose back."""
#     # Step 1: Load the file
#     score = converter.parse(input_file_path)

#     # Step 2: Detect original key
#     original_key = detect_key(score)

#     # Step 3: Transpose to C major or A minor
#     to_c_or_a_interval = get_transposition_interval(original_key)
#     transposed_score = transpose_score(score, to_c_or_a_interval)

#     # Step 4: Apply your processing function
#     processed_score = process_function(transposed_score)

#     # Step 5: Transpose back to original key
#     back_interval = to_c_or_a_interval.reverse()
#     final_score = transpose_score(processed_score, back_interval)

#     # Step 6: Write the result to a file
#     if output_file_path.endswith('.mid') or output_file_path.endswith('.midi'):
#         final_score.write('midi', fp=output_file_path)
#     else:
#         final_score.write('musicxml', fp=output_file_path)
