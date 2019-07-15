from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import pprint
from pickle import dump, load
from keras.utils.vis_utils import plot_model

def plot_acc(history_dict, epochs):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Testing acc')
    plt.title('Training and testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def parse_file(file, num_samples):
    eng_sentences = []
    fra_sentences = []
    eng_chars = set()
    fra_chars = set()

    for line in range(num_samples):

        split_line = str(file[line]).split('\t')
        eng_line = split_line[0]
        fra_line = '\t' + split_line[1] + '\n'  # '\t' = start token, '\n' = end token
        eng_sentences.append(eng_line)
        fra_sentences.append(fra_line)

        for char in eng_line:
            if char not in eng_chars:
                eng_chars.add(char)

        for char in fra_line:
            if char not in fra_chars:
                fra_chars.add(char)

    fra_chars = sorted(list(fra_chars))  # all possible characters per language
    eng_chars = sorted(list(eng_chars))

    return eng_sentences, eng_chars, fra_sentences, fra_chars

def create_vocabulary(chars):
    char2idx = {}  # vocabulary, all possible characters, maps character -> index
    idx2char = {}  # reverse vocabulary, for decoding

    for k, v in enumerate(chars):
        char2idx[v] = k
        idx2char[k] = v

    return char2idx, idx2char

def vectorize(num, eng_sentences, input_sequences, eng2idx, fra_sentences, output_sequences, fra2idx, target_data):
    # sentence vectorization: char -> one-hot
    for i in range(num):
        for k, char in enumerate(eng_sentences[i]):
            input_sequences[i, k, eng2idx[char]] = 1

        for k, char in enumerate(fra_sentences[i]):
            output_sequences[i, k, fra2idx[char]] = 1
            if k > 0:  # ahead by one timestep, without start token
                target_data[i, k - 1, fra2idx[char]] = 1

    return input_sequences, output_sequences, target_data

def decode_sequence(encoder_model, decoder_model, input_sequence, fra_chars, fra2idx, idx2fra, max_len_fra):
    # encode the input as state vectors
    states_value = encoder_model.predict(input_sequence)

    # generate empty target sequence of length 1
    target_sequence = np.zeros((1, 1, len(fra_chars)))
    # populate the first character of target sequence with the start character
    target_sequence[0, 0, fra2idx['\t']] = 1

    prediction = ''
    stop_condition = False

    while not stop_condition:
        output_token, h, c = decoder_model.predict(x=[target_sequence] + states_value)

        predicted_token_index = np.argmax(output_token[0, 0, :])
        predicted_char = idx2fra[predicted_token_index]
        prediction += predicted_char

        # exit condition, either hit max length or find stop character
        if (predicted_char == '\n') or (len(prediction) > max_len_fra):
            stop_condition = True

        target_sequence = np.zeros((1, 1, len(fra_chars)))
        target_sequence[0, 0, predicted_token_index] = 1

        states_value = [h, c]

    return prediction


# MAIN
# task: english -> french translator

# parse file
file = open('../data/translations.txt', encoding='utf-8').read().split('\n')
num_samples = 2000

# get data, prepare vocabulary
eng_sentences, eng_chars, fra_sentences, fra_chars = parse_file(file, num_samples)
eng2idx, idx2eng = create_vocabulary(eng_chars)
fra2idx, idx2fra = create_vocabulary(fra_chars)
max_len_eng = max([len(line) for line in eng_sentences])
max_len_fra = max([len(line) for line in fra_sentences])

# vectorize data
input_sequences = np.zeros(shape=(num_samples, max_len_eng, len(eng_chars)), dtype='float32')
output_sequences = np.zeros(shape=(num_samples, max_len_fra, len(fra_chars)), dtype='float32')
target_data = np.zeros(shape=(num_samples, max_len_fra, len(fra_chars)), dtype='float32')
input_sequences, output_sequences, target_data = vectorize(num_samples, eng_sentences, input_sequences, eng2idx,
                                                           fra_sentences, output_sequences, fra2idx, target_data)

print(fra_sentences[0])
print(fra2idx)
doutput = output_sequences[0]
dtarget = target_data[0]
dtest = (output_sequences[0])[1:]

for c in doutput:
    print(np.argmax(c), " ", end="", flush=True)
print("")
for c in dtarget:
    print(np.argmax(c), " ", end="", flush=True)
print("")
tmp = []
for c in dtest:
    tmp.append(np.argmax(c))
tmp = tmp + [0]
for c in tmp:
    print(c, " ", end="", flush=True)
exit()

# dump(input_sequences, open('data/input_sequences.pkl', 'wb'))
# input_sequences = load(open('data/input_sequences.pkl', 'rb'))

# model hyperparams
latent_size = 256  # number of units (output dimensionality)
batch_size = 64
epochs = 50

# encoder
encoder_input = Input(shape=(None, len(eng_chars)))
encoder_LSTM = LSTM(latent_size, return_state=True)
# returns last state (hidden state + cell state), discard encoder_outputs, only keep the states
# return state = returns the hidden state output and cell state for the last input time step
encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
encoder_states = [encoder_h, encoder_c]

# decoder
decoder_input = Input(shape=(None, len(fra_chars)))  # set up decoder, using encoder_states as initial state
decoder_LSTM = LSTM(latent_size, return_sequences=True, return_state=True)  # return state needed for inference
# return_sequence = returns the hidden state output for each input time step
decoder_outputs, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(fra_chars), activation='softmax')
decoder_out = decoder_dense(decoder_outputs)

# training
model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x=[input_sequences, output_sequences], y=target_data,
          batch_size=batch_size, epochs=epochs, validation_split=0.2)

# plot_model(model, to_file='model.png', show_shapes=True)
# model.save('data/model.h5')

history_dict = history.history
graph_epochs = range(1, epochs + 1)
plot_acc(history_dict, graph_epochs)

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

encoder_model = Model(encoder_input, encoder_states)

decoder_state_input_h = Input(shape=(latent_size,))
decoder_state_input_c = Input(shape=(latent_size,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]
decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input, initial_state=decoder_input_states)
decoder_states = [decoder_h, decoder_c]
decoder_out = decoder_dense(decoder_out)

decoder_model = Model(inputs=[decoder_input] + decoder_input_states, outputs=[decoder_out] + decoder_states)

# predictions
for index in range(10):
    input_sequence = input_sequences[index:index+1]
    prediction = decode_sequence(encoder_model, decoder_model, input_sequence,
                                       fra_chars, fra2idx, idx2fra, max_len_fra)

    print('-')
    print('Input sentence:', eng_sentences[index])
    print('Output sentence:', fra_sentences[index][1:-1])
    print('Decoded sentence:', prediction)
