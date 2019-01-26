from pickle import load
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt

def plot_acc(history_dict, epochs):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    fig = plt.figure()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Testing acc')
    plt.title('Training and testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# MAIN

stories = load(open('data/review_dataset.pkl', 'rb'))  # 1000 samples

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

for story in stories[:100]:
    input_text = story['story']
    for highlight in story['highlights']:
        target_text = highlight

    target_text = '\t' + target_text + '\n'  # "tab" = start token, "newline" = end token
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))  # all possible input characters
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# char2idx
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):  # vectorization: char -> one-hot
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.  # start and end token
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start token ("tab" = [1 0 0 0 0 ...])
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.  # only end token


# MODEL
batch_size = 32
epochs = 12
latent_dim = 128  # number of units (output dimensionality)

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
# returns last state (hidden state + cell state), discard encoder_outputs, only keep the states
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# set up decoder, using encoder_states as initial state
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# return full output sequences
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)  # return state needed for inference
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')  # predict next character in training
decoder_outputs = decoder_dense(decoder_outputs)

# encoder_input_data & decoder_input_data -> decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs, validation_split=0.2)

history_dict = history.history
print(round(max(history_dict['val_acc']), 3))
gprah_epochs = range(1, epochs + 1)
plot_acc(history_dict, gprah_epochs)
# model.save('data/model2.h5')

# INFERENCE
# define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# idx2char
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

"""
In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
"""

def decode_sequence(input_seq):
    # encode the input as state vectors
    states_value = encoder_model.predict(input_seq)
    # generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # populate the first character of target sequence with the start character
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # exit condition, either hit max length or find stop character
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))  # update the target sequence
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]  # update states

    return decoded_sentence


# predictions
for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
