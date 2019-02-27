import numpy

# predict (inference loop)
def decode_sequence(encoder_model, decoder_model, max_length, input_seq, char2idx, idx2char):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = numpy.zeros((1, 1, len(char2idx)))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, char2idx['\t']] = 1.  # 3D array, first element is start token

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = numpy.argmax(output_tokens[0, -1, :])  # greedy search
        sampled_char = idx2char[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = numpy.zeros((1, 1, len(char2idx)))
        target_seq[0, 0, sampled_token_index] = 1.  # previous character

        # Update states
        states_value = [h, c]

    return decoded_sentence