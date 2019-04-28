import os
import nltk
import codecs
import itertools
import numpy
from pprint import pprint
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding, BatchNormalization
from keras import optimizers
from keras.models import load_model
from pickle import dump, load
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from bert.tokenization import FullTokenizer


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
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
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
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
            "pooled_output"
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


def convert_sample(tokenizer, sample, max_seq_length):
    words = tokenizer.tokenize(sample)
    if len(words) > max_seq_length - 2:
        words = words[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in words:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
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


def convert_samples_to_features(tokenizer, samples, max_seq_length):
    input_ids, input_masks, segment_ids = [], [], []

    for sample in samples:
        input_id, input_mask, segment_id = convert_sample(tokenizer, sample, max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)

    return np.array(input_ids), np.array(input_masks), np.array(segment_ids)


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)


def seq2seq_architecture(latent_size, vocabulary_size):
    enc_in_id = tf.keras.layers.Input(shape=(None,), name="Encoder-Input-ids")  # TODO: max_seq_length on all Inputs?
    enc_in_mask = tf.keras.layers.Input(shape=(None,), name="Encoder-Input-Masks")
    enc_in_segment = tf.keras.layers.Input(shape=(None,), name="Encoder-Input-Segment-ids")
    bert_encoder_inputs = [enc_in_id, enc_in_mask, enc_in_segment]  # TODO: dimensions mismatch?

    encoder_embeddings = BertLayer(n_fine_tune_layers=3)(bert_encoder_inputs)
    encoder_embeddings = tf.keras.layers.BatchNormalization(name='Encoder-Batch-Normalization')(encoder_embeddings)
    _, state_h = tf.keras.layers.GRU(latent_size, return_state=True, name='Encoder-GRU')(encoder_embeddings)
    encoder_model = tf.keras.models.Model(inputs=bert_encoder_inputs, outputs=state_h, name='Encoder-Model')
    encoder_outputs = encoder_model(bert_encoder_inputs)

    dec_in_id = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-ids")
    dec_in_mask = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-Masks")
    dec_in_segment = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-Segment-ids")
    bert_decoder_inputs = [dec_in_id, dec_in_mask, dec_in_segment]

    decoder_embeddings = BertLayer(n_fine_tune_layers=3)(bert_decoder_inputs)
    decoder_embeddings = tf.keras.layers.BatchNormalization(name='Decoder-Batchnormalization-1')(decoder_embeddings)
    decoder_gru = tf.keras.layers.GRU(latent_size, return_state=True, return_sequences=True, name='Decoder-GRU')
    decoder_gru_outputs, _ = decoder_gru(decoder_embeddings, initial_state=encoder_outputs)
    decoder_outputs = tf.keras.layers.BatchNormalization(name='Decoder-Batchnormalization-2')(decoder_gru_outputs)
    decoder_outputs = tf.keras.layers.Dense(vocabulary_size, activation='softmax', name='Final-Output-Dense')(decoder_outputs)

    seq2seq_model = tf.keras.models.Model([bert_encoder_inputs, bert_decoder_inputs], decoder_outputs)
    seq2seq_model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['acc'])

    return seq2seq_model


# MAIN

sess = tf.Session()

titles, summaries, articles = read_data()
max_len_article = 20  # TODO
max_len_summary = 20  # TODO

tokenizer = create_tokenizer_from_hub_module(sess)
vocabulary_size = len(tokenizer.vocab)
print(vocabulary_size)

article_input_ids, article_input_masks, article_segment_ids = convert_samples_to_features(
    tokenizer, articles, max_len_article)

exit()

latent_size = 96
seq2seq_model = seq2seq_architecture(latent_size, vocabulary_size)
seq2seq_model.summary()

initialize_vars(sess)

