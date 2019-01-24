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
for story in stories[:200]:
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

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(  # char2idx
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

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


# model
batch_size = 32
epochs = 10
latent_dim = 64  # number of units (output dimensionality)


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

# inference
