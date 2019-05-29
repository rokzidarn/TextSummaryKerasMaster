import os
import codecs
import numpy
from pprint import pprint
import matplotlib.pyplot as plt
from keras.models import load_model
from pickle import dump, load
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from bert.tokenization import FullTokenizer


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=3, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1",
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables
        # trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name and not "/pooler/" in var.name]
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [tf.keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "sequence_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size


def plot_acc(history_dict, epochs):
    acc = history_dict['acc']

    fig = plt.figure()
    plt.plot(epochs, acc, 'r')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    fig.savefig('data/models/bert_gru_s2q.png')


def read_data():
    summaries = []
    articles = []
    titles = []

    ddir = 'data/test/'
    summary_files = os.listdir(ddir+'summaries/')
    for file in summary_files:
        f = codecs.open(ddir+'summaries/'+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        summaries.append(' '.join(tmp))
        titles.append(file[:-4])

    article_files = os.listdir(ddir+'articles/')
    for file in article_files:
        f = codecs.open(ddir+'articles/'+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        articles.append(' '.join(tmp))

    return titles, summaries, articles


def create_tokenizer_from_hub_module(sess):
    bert_module = hub.Module("https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1")
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def tokenize_samples(tokenizer, samples):
    words = []

    for sample in samples:
        words.append(tokenizer.tokenize(sample))

    max_seq_length = len(max(words, key=len))

    return words, max_seq_length


def convert_sample(words, max_seq_length):
    if len(words) > max_seq_length - 2:
        words = words[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")  # 101
    segment_ids.append(0)

    for token in words:
        tokens.append(token)
        segment_ids.append(0)  # TODO: check segments, sentence splitting

    tokens.append("[SEP]")  # 102
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # the mask has 1 for real tokens and 0 for padding tokens, only real tokens are attended to
    input_mask = [1] * len(input_ids)

    # zero-pad up to the sequence length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids


def vectorize_features(samples, max_seq_length):
    input_ids, input_masks, segment_ids = [], [], []

    for sample in samples:
        input_id, input_mask, segment_id = convert_sample(sample, max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)

    return np.array(input_ids), np.array(input_masks), np.array(segment_ids)


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)


def seq2seq_architecture(latent_size, vocabulary_size, max_len_article, max_len_summary):
    enc_in_id = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-ids")  # None
    enc_in_mask = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-Masks")
    enc_in_segment = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-Segment-ids")
    bert_encoder_inputs = [enc_in_id, enc_in_mask, enc_in_segment]

    encoder_embeddings = BertLayer(name='Encoder-Bert-Layer')(bert_encoder_inputs)
    encoder_embeddings = tf.keras.layers.BatchNormalization(name='Encoder-Batch-Normalization')(encoder_embeddings)
    _, state_h = tf.keras.layers.GRU(latent_size, return_state=True, name='Encoder-GRU')(encoder_embeddings)
    encoder_model = tf.keras.models.Model(inputs=bert_encoder_inputs, outputs=state_h, name='Encoder-Model')
    encoder_outputs = encoder_model(bert_encoder_inputs)

    dec_in_id = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-ids")
    dec_in_mask = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-Masks")
    dec_in_segment = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-Segment-ids")
    bert_decoder_inputs = [dec_in_id, dec_in_mask, dec_in_segment]

    decoder_embeddings = BertLayer(name='Decoder-Bert-Layer')(bert_decoder_inputs)
    decoder_embeddings = tf.keras.layers.BatchNormalization(name='Decoder-Batchnormalization-1')(decoder_embeddings)
    decoder_gru = tf.keras.layers.GRU(latent_size, return_state=True, return_sequences=True, name='Decoder-GRU')
    decoder_gru_outputs, _ = decoder_gru(decoder_embeddings, initial_state=encoder_outputs)
    decoder_outputs = tf.keras.layers.BatchNormalization(name='Decoder-Batchnormalization-2')(decoder_gru_outputs)
    decoder_outputs = tf.keras.layers.Dense(vocabulary_size, activation='softmax', name='Final-Output-Dense')(decoder_outputs)

    seq2seq_model = tf.keras.models.Model(inputs=[enc_in_id, enc_in_mask, enc_in_segment,
                                                  dec_in_id, dec_in_mask, dec_in_segment], outputs=decoder_outputs)
    seq2seq_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy',
                          metrics=['acc'])

    return seq2seq_model


def inference(model, latent_dim):
    encoder_model = model.get_layer('Encoder-Model')

    # latent_dim = model.get_layer('Decoder-Bert-Layer').output_shape[-1]  # 768
    dec_in_id = model.get_layer("Decoder-Input-ids").input
    dec_in_mask = model.get_layer("Decoder-Input-Masks").input
    dec_in_segment = model.get_layer("Decoder-Input-Segment-ids").input
    bert_decoder_inputs = [dec_in_id, dec_in_mask, dec_in_segment]
    decoder_embeddings = model.get_layer('Decoder-Bert-Layer')(bert_decoder_inputs)

    decoder_embeddings = model.get_layer('Decoder-Batchnormalization-1')(decoder_embeddings)
    gru_inference_state_input = tf.keras.layers.Input(shape=(latent_dim,), name='hidden_state_input')
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([decoder_embeddings, gru_inference_state_input])
    decoder_outputs = model.get_layer('Decoder-Batchnormalization-2')(gru_out)
    dense_out = model.get_layer('Final-Output-Dense')(decoder_outputs)
    decoder_model = tf.keras.models.Model([dec_in_id, dec_in_mask, dec_in_segment, gru_inference_state_input],
                                          [dense_out, gru_state_out])

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, inputs, max_len, tokenizer):
    input_ids, input_masks, segment_ids = inputs
    states_value = encoder_model.predict([input_ids, input_masks, segment_ids])
    target_sequence = numpy.array(tokenizer.convert_tokens_to_ids(["[CLS]"])).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, state = decoder_model.predict([target_sequence, states_value])

        predicted_word_index = numpy.argmax(candidates)
        predicted_word = tokenizer.convert_ids_to_tokens([predicted_word_index])
        prediction.append(predicted_word)

        if (predicted_word == "[SEP]") or (len(prediction) > max_len):
            stop_condition = True

        states_value = state
        target_sequence = numpy.array(predicted_word_index).reshape(1, 1)

    return prediction[:-1]


# MAIN

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sess = tf.Session()
tokenizer = create_tokenizer_from_hub_module(sess)
"""
words = tokenizer.tokenize("Pozdravljeni, gospod Rok. Dober dan tudi Vam!")
idx = tokenizer.convert_tokens_to_ids(words)
repeat = tokenizer.convert_ids_to_tokens(idx)
first = tokenizer.convert_ids_to_tokens([idx[0]])
print(len(words), idx, repeat, first)

special = "[CLS]"
sidx = tokenizer.convert_tokens_to_ids([special])
convert = tokenizer.convert_ids_to_tokens(sidx)
print(sidx, convert)

vectorized = convert_sample(words, 30)
print(vectorized)
"""

titles, summaries, articles = read_data()
article_tokens, max_len_article = tokenize_samples(tokenizer, articles)
summary_tokens, max_len_summary = tokenize_samples(tokenizer, summaries)
vocabulary_size = len(tokenizer.vocab)

article_input_ids, article_input_masks, article_segment_ids = vectorize_features(article_tokens, max_len_article)
summary_input_ids, summary_input_masks, summary_segment_ids = vectorize_features(summary_tokens, max_len_summary)

target_input_ids, target_masks, target_segment_ids = [], [], []
# TODO: https://stackoverflow.com/questions/50530100/keras-lstm-multi-output-model-predict-two-features-time-series

for summary_input_id in summary_input_ids:
    target_input_id = numpy.append(summary_input_id[1:], 0)
    target_input_ids.append(target_input_id)

latent_size = 32
batch_size = 1
epochs = 5

seq2seq_model = seq2seq_architecture(latent_size, vocabulary_size, max_len_article, max_len_summary)
seq2seq_model.summary()

initialize_vars(sess)

seq2seq_model.fit([article_input_ids, article_input_masks, article_segment_ids,
                   summary_input_ids, summary_input_masks, summary_segment_ids],
                  numpy.expand_dims(target_input_ids, -1), epochs=epochs, batch_size=batch_size)

encoder_model, decoder_model = inference(seq2seq_model, latent_size)

for i in range(5):
    inputs = article_input_ids[i:i+1], article_input_masks[i:i+1], article_segment_ids[i:i+1]
    prediction = predict_sequence(encoder_model, decoder_model, inputs, max_len_summary, tokenizer)

    print('-')
    print('Summary:', summary_tokens[i])
    print('Prediction:', prediction)
