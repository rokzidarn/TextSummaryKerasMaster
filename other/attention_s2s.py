import os
import nltk
import codecs
import itertools
import numpy
import matplotlib.pyplot as plt
import rouge
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Concatenate, BatchNormalization
from tensorflow.python.keras.models import Model
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Concatenate, BatchNormalization


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]


def read_data_train():
    summaries = []
    articles = []
    titles = []

    ddir = '../data/test/'
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


def clean_data(text):
    tokens = nltk.word_tokenize(text)
    exclude_list = [',', '.', '(', ')', '>', '>', '»', '«', ':', '–', '-', '+', '–', '--', '/', '|', '“', '”', '•',
                    '``', '\"', "\'\'", '?', '!', ';', '*', '†', '[', ']', '%', '—', '°', '…', '=', '#', '<', '>']
    # keeps dates and numbers, excludes the rest
    clean_tokens = [e.lower() for e in tokens if e not in exclude_list]

    return clean_tokens


def build_vocabulary(tokens):
    fdist = nltk.FreqDist(tokens)

    word2idx = {w: (i + 4) for i, (w, _) in enumerate(fdist.most_common())}  # vocabulary, 'word' -> 11
    word2idx['<PAD>'] = 0  # padding
    word2idx['<START>'] = 1  # start token
    word2idx['<END>'] = 2  # end token
    word2idx['<UNK>'] = 3  # unknown token

    idx2word = {v: k for k, v in word2idx.items()}  # inverted vocabulary, for decoding, 11 -> 'word'

    return fdist, word2idx, idx2word


def pre_process(texts, word2idx):  # vectorizes texts, array of tokens (words) -> array of ints (word2idx)
    vectorized_texts = []

    for text in texts:
        text_vector = [word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in text]
        text_vector.insert(0, word2idx['<START>'])  # add <START> and <END> tokens to each summary/article
        text_vector.append(word2idx['<END>'])
        vectorized_texts.append(text_vector)

    return vectorized_texts


def process_targets(summaries_train, word2idx):
    tmp_vectors = pre_process(summaries_train, word2idx)  # same as summaries_vectors, but with delay
    target_vectors = []  # ahead by one timestep, without start token
    for tmp in tmp_vectors:
        tmp.append(word2idx['<PAD>'])  # added <PAD>, so the dimensions match
        target_vectors.append(tmp[1:])

    return target_vectors


def post_process(predictions, idx2word):  # transform array of ints (idx2word) -> array of tokens (words)
    predicted_texts = []

    for output in predictions:
        predicted_words = [idx2word[idx] if idx in idx2word else "<UNK>" for idx in output]
        predicted_texts.append(predicted_words)

    return predicted_texts


def plot_training(history_dict, epochs):
    loss = history_dict['loss']

    fig = plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    fig.savefig('../data/models/attention_seq2seq.png')


def seq2seq_architecture(latent_size, embedding_size, vocabulary_size, max_len_article, batch_size, epochs, sess):
    # encoder
    encoder_inputs = Input(shape=(max_len_article,))
    encoder_embeddings = Embedding(vocabulary_size, embedding_size, trainable=True)(encoder_inputs)
    encoder_embeddings = BatchNormalization()(encoder_embeddings)
    encoder_outputs, e_state_h, e_state_c = LSTM(latent_size, return_state=True, return_sequences=True,
                                                 dropout=0.4, recurrent_dropout=0.4)(encoder_embeddings)
    # decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embeddings_layer = Embedding(vocabulary_size, embedding_size, trainable=True)
    decoder_embeddings = decoder_embeddings_layer(decoder_inputs)

    decoder_batchnorm_layer = BatchNormalization()
    encoder_embeddings = decoder_batchnorm_layer(decoder_embeddings)
    decoder_lstm = LSTM(latent_size, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.2)
    decoder_outputs, d_state_h, d_state_c = decoder_lstm(decoder_embeddings, initial_state=[e_state_h, e_state_c])

    attention_layer = AttentionLayer(name='Attention-Layer')
    attention_out, attention_states = attention_layer([encoder_outputs, decoder_outputs])
    decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='Concat-Layer')([decoder_outputs, attention_out])

    decoder_dense = TimeDistributed(Dense(vocabulary_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    seq2seq_model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy',
                          metrics=['sparse_categorical_accuracy'])

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)  # stop training once validation loss increases
    seq2seq_model.summary()
    initialize_vars(sess)
    history = seq2seq_model.fit([X_article, X_summary], numpy.expand_dims(Y_target, -1),
                                batch_size=batch_size, epochs=epochs)

    f = open("../data/models/attention_results.txt", "w", encoding="utf-8")
    f.write("Attention \n layers: 1 \n latent size: " + str(latent_size) + "\n embeddings size: " + str(embedding_size) + "\n")
    f.close()

    history_dict = history.history
    graph_epochs = range(1, epochs + 1)
    plot_training(history_dict, graph_epochs)

    # inference
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, e_state_h, e_state_c])

    decoder_state_input_h = Input(shape=(latent_size,))
    decoder_state_input_c = Input(shape=(latent_size,))
    decoder_hidden_state_input = Input(shape=(max_len_article, latent_size))

    decoder_embeddings_inf = decoder_embeddings_layer(decoder_inputs)
    decoder_embeddings_inf = decoder_batchnorm_layer(decoder_embeddings_inf)
    decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(decoder_embeddings_inf,
                                                                 initial_state=[decoder_state_input_h, decoder_state_input_c])

    attn_out_inf, attn_states_inf = attention_layer([decoder_hidden_state_input, decoder_outputs_inf])
    decoder_inf_concat = Concatenate(axis=-1, name='Concat')([decoder_outputs_inf, attn_out_inf])
    decoder_outputs_inf = decoder_dense(decoder_inf_concat)

    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs_inf] + [state_h_inf, state_c_inf])

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len):
    encoder_out, out_h, out_c = encoder_model.predict(input_sequence)
    target_sequence = numpy.array(word2idx['<START>']).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, h, c = decoder_model.predict([target_sequence] + [encoder_out, out_h, out_c])

        predicted_word_index = numpy.argmax(candidates)  # greedy search
        predicted_word = idx2word[predicted_word_index]
        prediction.append(predicted_word)

        if (predicted_word == '<END>') or (len(prediction) > max_len):
            stop_condition = True

        out_h, out_c = h, c
        target_sequence = numpy.array(predicted_word_index).reshape(1, 1)  # previous character

    return prediction[:-1]


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def evaluate(encoder_model, decoder_model, titles_train, summaries_train, X_article, word2idx, idx2word, max_length_summary):
    predictions = []

    # testing
    for index in range(len(titles_train)):
        input_sequence = X_article[index:index+1]
        prediction = predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word,
                                      max_length_summary)

        predictions.append(prediction)
        print(prediction)
        # f = open("../data/bert/predictions/" + titles_train[index] + ".txt", "w", encoding="utf-8")
        # f.write(str(prediction))
        # f.close()

    # evaluation using ROUGE
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=3,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=False,
                            apply_best=True,
                            alpha=0.5,  # default F1 score
                            weight_factor=1.2,
                            stemming=True)

    all_hypothesis = [' '.join(prediction) for prediction in predictions]
    all_references = [' '.join(summary) for summary in summaries_train]
    scores = evaluator.get_scores(all_hypothesis, all_references)

    f = open("../data/models/attention_results.txt", "a", encoding="utf-8")
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        result = prepare_results(metric, results['p'], results['r'], results['f'])
        print(result)
        f.write('\n' + result)
    f.close()


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)


# MAIN
# https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/

sess = tf.Session()

# 1D array, each element is string of sentences, separated by newline
titles_train, summaries_train, articles_train = read_data_train()

# 2D array, array of summaries/articles, sub-arrays of words
summaries_train = [clean_data(summary) for summary in summaries_train]
articles_train = [clean_data(article) for article in articles_train]

max_length_summary = len(max(summaries_train, key=len)) + 2  # with <START> and <END> tokens added
max_length_article = len(max(articles_train, key=len)) + 2

all_tokens = list(itertools.chain(*summaries_train)) + list(itertools.chain(*articles_train))
fdist, word2idx, idx2word = build_vocabulary(all_tokens)
vocabulary_size = len(word2idx.items())  # with <PAD>, <START>, <END>, <UNK> tokens

print("DATASET DATA:")
print('Dataset size (number of summary-article pairs): ', len(summaries_train))
print('Max lengths of summary/article in dataset: ', max_length_summary, '/', max_length_article)
print('Vocabulary size (number of all possible words): ', vocabulary_size)
print('Most common words: ', fdist.most_common(10))
print('Vocabulary (word -> index): ', {k: word2idx[k] for k in list(word2idx)[:10]})
print('Inverted vocabulary (index -> word): ', {k: idx2word[k] for k in list(idx2word)[:10]})

# 2D array, array of summaries/articles, sub-arrays of indexes (int)
summaries_vectors = pre_process(summaries_train, word2idx)
articles_vectors = pre_process(articles_train, word2idx)
target_vectors = process_targets(summaries_train, word2idx)

X_summary = pad_sequences(summaries_vectors, maxlen=max_length_summary, padding='post')
X_article = pad_sequences(articles_vectors, maxlen=max_length_article, padding='post')
Y_target = pad_sequences(target_vectors, maxlen=max_length_summary, padding='post')

# model hyper parameters
latent_size = 128  # number of units (output dimensionality)
embedding_size = 96  # word vector size
batch_size = 1
epochs = 12

# training
encoder_model, decoder_model = seq2seq_architecture(latent_size, embedding_size, vocabulary_size,
                                                    max_length_article, batch_size, epochs, sess)

# testing
evaluate(encoder_model, decoder_model, titles_train, summaries_train, X_article, word2idx, idx2word, max_length_summary)