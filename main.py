import numpy as np
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt

def generate_sequence(length, n_unique):
    return [randint(1, n_unique-1) for _ in range(length)]

def generate_dataset(input_sample_size, output_sample_size, sample_token_size, n_samples):
    X1, X2, y = list(), list(), list()

    for _ in range(n_samples):
        source = generate_sequence(input_sample_size, sample_token_size)  # input sequence
        target = source[:output_sample_size]  # output sequence
        target.reverse()

        # add start token to target sequence, 1 timestep ahead, last token discarded
        target_in = [0] + target[:-1]

        # one-hot encode: 3 -> [0 0 0 1 0 ...]
        src_encoded = to_categorical([source], num_classes=sample_token_size)
        tar_encoded = to_categorical([target], num_classes=sample_token_size)
        tar_in_encoded = to_categorical([target_in], num_classes=sample_token_size)

        X1.append(src_encoded)
        X2.append(tar_in_encoded)
        y.append(tar_encoded)

    X1 = np.squeeze(array(X1), axis=1)  # removes single-dimensional entries (2,1,6,10) -> (2,6,10)
    X2 = np.squeeze(array(X2), axis=1)
    y = np.squeeze(array(y), axis=1)

    return X1, X2, y

def define_models(n_input, n_output, latent_units):
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(latent_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(latent_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # inference
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_units,))
    decoder_state_input_c = Input(shape=(latent_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def predict_sequence(inf_encoder_model, inf_decoder_model, input_sequence, output_sample_size, sample_token_size):
    state = inf_encoder_model.predict(input_sequence)  # get input through encoder

    # generate start token, first input to decoder
    target_seq = array([0.0 for _ in range(sample_token_size)]).reshape(1, 1, sample_token_size)
    prediction = list()  # prediction storage, updated token by token

    for t in range(output_sample_size):
        # predict next char
        yhat, h, c = inf_decoder_model.predict([target_seq] + state)
        predicted_token = argmax(yhat[0, 0, :])
        prediction.append(predicted_token)  # possibility distribution for each possible token

        state = [h, c]  # update state
        target_seq = yhat  # update target sequence, next input for decoder

    return array(prediction)

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
# SEQ2SEQ

input_sample_size = 6  # array length of sample: [x1 x2 x3 x4 x5 x6]
output_sample_size = 3
sample_token_size = 9 + 1  # possible classes, tokens [1,9]: e.g. [4, 5, 1, 9, 3, 3]
dataset_size = 6000
# task: [4, 5, 1, 9, 3, 3] -> [1, 5, 4]

test_sample = generate_sequence(input_sample_size, sample_token_size)  # 0-9 numbers as possible tokens, array length 6
X1, X2, y = generate_dataset(input_sample_size, output_sample_size, sample_token_size, dataset_size)

latent_size = 128  # number of LSTM cells
epochs = 4
train_model, inf_encoder_model, inf_decoder_model = define_models(sample_token_size, sample_token_size, latent_size)
train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = train_model.fit([X1, X2], y, epochs=epochs, validation_split=0.1)

history_dict = history.history
gprah_epochs = range(1, epochs + 1)
plot_acc(history_dict, gprah_epochs)

# validation
total, correct = 100, 0
for _ in range(total):
    X1, X2, y = generate_dataset(input_sample_size, output_sample_size, sample_token_size, 1)
    target = [argmax(vector) for vector in y[0]]
    prediction = predict_sequence(inf_encoder_model, inf_decoder_model, X1, output_sample_size, sample_token_size)
    if array_equal(target, prediction):
        correct += 1

print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

# check
for _ in range(5):
    X1, X2, y = generate_dataset(input_sample_size, output_sample_size, sample_token_size, 1)
    input = [argmax(vector) for vector in X1[0]]
    target = [argmax(vector) for vector in y[0]]
    prediction = predict_sequence(inf_encoder_model, inf_decoder_model, X1, output_sample_size, sample_token_size)
    print('X=%s y=%s, prediction=%s' % (input, target, prediction))
