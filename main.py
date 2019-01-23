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
# number of units (output dimensionality), returns last state (hidden state + cell state)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder
decoder_inputs = Input(shape=(None, max_length))
decoder_lstm = LSTM(512, return_sequences=True, return_state=True)  # returns full sequence
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(max_length, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
#print(model.summary())

history = model.fit([encoder_data, decoder_data], target_data,
          batch_size=32, epochs=100, validation_split=0.2)

history_dict = history.history
print(round(max(history_dict['val_acc']), 3))
