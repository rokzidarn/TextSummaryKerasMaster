from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, BatchNormalization
from keras import optimizers
import numpy

encoder_input_data = numpy.array([
    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
])

decoder_input_data = numpy.array([
    [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
])

decoder_target_data = numpy.array([
    [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
])

doc_length = 1
num_encoder_tokens = num_decoder_tokens = 2
embedding_dim = 4

# encoder model
encoder_inputs = Input(shape=(doc_length,), name='Encoder-Input')
enc_emb = Embedding(num_encoder_tokens, embedding_dim, name='Body-Word-Embedding', mask_zero=False)(encoder_inputs)
enc_bn = BatchNormalization(name='Encoder-Batchnorm-1')(enc_emb)

encoder_lstm = LSTM(128, name='Encoder-Intermediate-LSTM', return_sequences=True)(enc_bn)
_, state_h = LSTM(128, return_state=True, name='Encoder-Last-LSTM')(encoder_lstm)
encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)

# decoder model
decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing
dec_emb = Embedding(num_decoder_tokens, embedding_dim, name='Decoder-Word-Embedding', mask_zero=False)(decoder_inputs)
dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

decoder_lstm = LSTM(128, return_state=True, return_sequences=True, name='Decoder-GRU')
decoder_lstm_output, _ = decoder_lstm(dec_bn, initial_state=seq2seq_encoder_out)
x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_lstm_output)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='Final-Output-Dense')
decoder_outputs = decoder_dense(x)

# seq2seq model
seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
seq2seq_model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')
seq2seq_model.summary()

# train
history = seq2seq_model.fit([encoder_input_data, decoder_input_data], numpy.expand_dims(decoder_target_data, -1),
          batch_size=32, epochs=20, validation_split=0.12,)

# predict
