import numpy
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# DATA
encoder_data = numpy.array([
    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
])

decoder_data = numpy.array([
    [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
    [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
    [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
])

target_data = numpy.array([  # 1 timestep offset from decoder_data
    [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
    [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
])

num_samples = encoder_data.shape[0]
sample_size = encoder_data.shape[1]
max_length = encoder_data.shape[2]

# MODEL

# encoder
encoder_inputs = Input(shape=(None, max_length))
encoder_lstm = LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder
decoder_inputs = Input(shape=(None, max_length))
decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(max_length, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# s2s
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit([encoder_data, decoder_data], target_data,
          batch_size=32, epochs=100, validation_split=0.2)

history_dict = history.history
print(round(max(history_dict['val_acc']), 3))

# inference setup
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# predict (inference loop)
def decode_sequence(input_seq, char2idx, idx2char):
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
