import os
import io
import codecs
import itertools
import re
import copy
import nltk
import rouge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Input, LSTM, Embedding, Dense, BatchNormalization, TimeDistributed
from tensorflow.python.keras.models import Model


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention
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


def read_data():
    summaries = []
    articles = []
    titles = []

    ddir = 'data/sta/'

    article_files = os.listdir(ddir + 'articles/')
    for file in article_files:
        f = codecs.open(ddir + 'articles/' + file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        articles.append(' '.join(tmp))

    summary_files = os.listdir(ddir + 'summaries/')
    for file in summary_files:
        f = codecs.open(ddir + 'summaries/' + file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        summaries.append(' '.join(tmp))
        titles.append(file[:-4])

    return titles, articles, summaries


def load_embeddings():
    fin = io.open('data/fasttext/cc.sl.300.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    embeddings_index = {}
    words = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        words.append(word)
        coefs = np.asarray(tokens[1:], dtype='float32')
        embeddings_index[word] = coefs
    return embeddings_index, n, d, words


def plot_loss(history_dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = list(range(1, len(loss) + 1))

    fig = plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('data/models/final_train.png')

    fig = plt.figure()
    plt.plot(epochs, val_loss, 'r')
    plt.title('Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('data/models/final_valid.png')


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1',
                                                                 100.0 * f)


def clean_data(data, threshold):
    cleaned = []

    for text in data:
        tokens = nltk.word_tokenize(text)
        exclude_list = [',', '.', '(', ')', '>', '<', '»', '«', ':', '–', '-', '+', '–', '--', '/', '|', '“', '”', '•',
                        '``', '\"', "\'\'", '?', '!', ';', '*', '†', '[', ']', '%', '—', '°', '…', '=', '#', '&', '\'',
                        '$', '...', '}', '{', '„', '@', '', '//', '½', '***', '’', '·', '©']

        # keeps dates and numbers, excludes the rest
        excluded = [e for e in tokens if e not in exclude_list]  # lower()

        # remove decimals, html texts
        clean = []
        for e in excluded:
            if not any(re.findall(r'fig|pic|\+|\,|\.|å', e, re.IGNORECASE)):
                clean.append(e)

        cleaned.append(clean[:threshold])

    return cleaned


def analyze_data(data, show_plot=False):
    lengths = [len(text)+2 for text in data]  # add 2 because of special tokens <START>, <END>
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = int(round(sum(lengths) / len(lengths)))

    if show_plot:
        samples = list(range(1, len(lengths) + 1))
        fig, ax = plt.subplots()
        data_line = ax.plot(samples, lengths, label='Data', marker='o')
        min_line = ax.plot(samples, [min_len] * len(lengths), label='Min', linestyle='--')
        max_line = ax.plot(samples, [max_len] * len(lengths), label='Max', linestyle='--')
        avg_line = ax.plot(samples, [avg_len] * len(lengths), label='Avg', linestyle='--')
        legend = ax.legend(loc='upper right')
        plt.show()

    return min_len, max_len, avg_len


def build_vocabulary(tokens, embedding_words, write_dict=False):
    fdist = nltk.FreqDist(tokens)
    # fdist.plot(50)  # fdist.pprint(maxlen=50)

    all = fdist.most_common()  # unique_words = fdist.hapaxes()
    sub_all = [element for element in all if element[1] > 25]  # cut vocabulary

    embedded = []  # exclude words that are not in embedding matrix
    for element in sub_all:
        w, r = element
        if w in embedding_words:
            embedded.append(element)

    word2idx = {w: (i + 4) for i, (w, _) in enumerate(embedded)}
    word2idx['<PAD>'] = 0  # padding
    word2idx['<START>'] = 1  # start token
    word2idx['<END>'] = 2  # end token
    word2idx['<UNK>'] = 3  # unknown token

    idx2word = {v: k for k, v in word2idx.items()}  # inverted vocabulary, for decoding, 11 -> 'word'

    if write_dict:
        print('All vocab:', len(all))
        print('Sub vocab: ', len(sub_all))
        print('Embedding vocab: ', len(embedding_words))
        print('Final vocab: ', len(embedded))

        f = open("data/models/data_dict.txt", "w", encoding='utf-8')
        for k, v in word2idx.items():
            f.write(k + '\n')
        f.close()

    return fdist, word2idx, idx2word


def count_padding_unknown(article_inputs, summary_inputs):
    article_unk = []
    summary_unk = []
    article_pad = []
    summary_pad = []

    for article in article_inputs:
        article = list(article)
        article_unk.append(article.count(3)/len(article))  # <UNK> = 3
        article_pad.append(article.count(0)/len(article))  # <PAD> = 0

    for summary in summary_inputs:
        summary = list(summary)
        summary_unk.append(summary.count(3)/len(summary))
        summary_pad.append(summary.count(0)/len(summary))

    return article_unk, summary_unk, article_pad, summary_pad


def pre_process(texts, word2idx, reverse):
    vectorized_texts = []

    for text in texts:  # vectorizes texts, array of tokens (words) -> array of ints (word2idx)
        text_vector = [word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in text]
        text_vector.insert(0, word2idx['<START>'])  # add <START> and <END> tokens to each summary/article
        text_vector.append(word2idx['<END>'])
        if reverse:
            vectorized_texts.append(list(reversed(text_vector)))
        else:
            vectorized_texts.append(text_vector)

    return vectorized_texts


def process_targets(summaries, word2idx):
    tmp_inputs = pre_process(summaries, word2idx, False)  # same as summaries_vectors, but with delay
    target_inputs = []  # ahead by one timestep, without start token
    for tmp in tmp_inputs:
        tmp.append(0)  # added 0 for padding, so the dimensions match
        target_inputs.append(tmp[1:])

    return target_inputs


def seq2seq_architecture(latent_size, vocabulary_size, max_len_article, embedding_matrix, batch_size, epochs,
                         train_article, train_summary, train_target):
    """
    The RNN function takes the current RNN state and a word vector and produces a
    subsequent RNN state that “encodes” the sentence so far.
    input_shape = (batch_size, time_steps, seq_len)
    LSTM -> (batch_size, latent_size)
    LSTM + return_sequences -> (batch_size, time_steps, units)

    output, state_h, state_c = LSTM(units, return_state=True)  # h<T>, h<T>, c<T>
    outputs, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)  # h<1...T>, h<T>, c<T>
    output, state_h = GRU(units, return_state=True)  # h<T>, h<T>
    outputs, state_h = GRU(units, return_sequences=True, return_state=True)  # h<1...T>, h<T>

    Each LSTM cell will output one hidden state h for each input.
    Stacking RNN, the former RNN layer or layers should set return_sequences to True so that the
    following RNN layer or layers can have the full sequence as input.
    LSTM with return_sequences=True returns the hidden state of the LSTM for every timestep in the input to the LSTM.
    For example, if the input batch is (samples, timesteps, dims), then the call LSTM(units, return_sequences=True)
    will generate output of dimensions (samples, timesteps, units).
    However, if you also want the final internal cell state of LSTM,
    use LSTM(units, return_sequence=True, return_state=True)
    In LSTMs return_sequences returns the states of the neurons at each timestep,
    return_states returns the internal cell states, which are responsible for memory control.
    """

    # encoder
    encoder_inputs = Input(shape=(max_len_article,), name='Encoder-Input')
    encoder_embeddings = Embedding(vocabulary_size, 300, weights=[embedding_matrix], trainable=False, mask_zero=False,
                                   name='Encoder-Word-Embedding')
    norm_encoder_embeddings = BatchNormalization(name='Encoder-Batch-Normalization')
    encoder_lstm_1 = LSTM(latent_size, name='Encoder-LSTM-1', return_sequences=True, return_state=True,
                          dropout=0.2, recurrent_dropout=0.2)
    encoder_lstm_2 = LSTM(latent_size, name='Encoder-LSTM-2', return_sequences=True, return_state=True,
                          dropout=0.2, recurrent_dropout=0.2)
    # the sequence of the last layer is not returned because we want a single vector that stores everything

    e = encoder_embeddings(encoder_inputs)
    e = norm_encoder_embeddings(e)
    e, e_state_h_1, e_state_c_1 = encoder_lstm_1(e)
    # returns last state (hidden state + cell state), discard encoder_outputs, only keep the states
    # return state = returns the hidden state output and cell state for the last input time step
    encoder_outputs, e_state_h_2, e_state_c_2 = encoder_lstm_2(e)
    # the encoded fix-sized vector which seq2seq is all about

    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, e_state_h_2, e_state_c_2,
                                                          e_state_h_2, e_state_c_2])

    # decoder
    decoder_inputs = Input(shape=(None,), name='Decoder-Input')
    decoder_embeddings = Embedding(vocabulary_size, 300, weights=[embedding_matrix], trainable=False, mask_zero=False,
                                   name='Decoder-Word-Embedding')
    norm_decoder_embeddings = BatchNormalization(name='Decoder-Batch-Normalization-1')
    decoder_lstm_1 = LSTM(latent_size, name='Decoder-LSTM-1', return_sequences=True, return_state=True,
                          dropout=0.2, recurrent_dropout=0.2)
    decoder_lstm_2 = LSTM(latent_size, name='Decoder-LSTM-2', return_sequences=True, return_state=True,
                          dropout=0.2, recurrent_dropout=0.2)
    norm_decoder = BatchNormalization(name='Decoder-Batch-Normalization-2')
    attention_layer = AttentionLayer(name='Attention-Layer')
    decoder_dense = TimeDistributed(Dense(vocabulary_size, activation='softmax'), name="Final-Output-Dense")

    d = decoder_embeddings(decoder_inputs)
    d = norm_decoder_embeddings(d)  # set up decoder, using encoder_states as initial state
    d, d_state_h_1, d_state_c_1 = decoder_lstm_1(d, initial_state=[e_state_h_1, e_state_c_1])
    # return state needed for inference
    # return_sequence = returns the hidden state output for each input time step
    decoder_outputs, d_state_h_2, d_state_c_2 = decoder_lstm_2(d, initial_state=[e_state_h_2, e_state_c_2])
    decoder_outputs = norm_decoder(decoder_outputs)
    attention_out, attention_states = attention_layer([encoder_outputs, decoder_outputs])
    decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='Concat-Layer')([decoder_outputs, attention_out])
    decoder_outputs = decoder_dense(decoder_concat_input)

    seq2seq_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    seq2seq_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',
                          metrics=['sparse_categorical_accuracy'])
    seq2seq_model.summary()

    classes = [item for sublist in train_summary.tolist() for item in sublist]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(classes), classes)

    e_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min', restore_best_weights=True)
    history = seq2seq_model.fit(x=[train_article, train_summary], y=np.expand_dims(train_target, -1),
                                batch_size=batch_size, epochs=epochs, validation_split=0.1,
                                callbacks=[e_stopping], class_weight=class_weights)

    f = open("data/models/final_results.txt", "w", encoding="utf-8")
    f.write("Final LSTM \n layers: 2 \n latent size: " + str(latent_size) + "\n vocab size: " +
            str(vocabulary_size) + "\n")
    f.close()

    history_dict = history.history
    plot_loss(history_dict)

    # inference
    decoder_initial_state_a1 = Input(shape=(max_len_article, latent_size), name='Decoder-Init-A1')
    decoder_initial_state_h1 = Input(shape=(latent_size,), name='Decoder-Init-H1')
    decoder_initial_state_c1 = Input(shape=(latent_size,), name='Decoder-Init-C1')
    decoder_initial_state_h2 = Input(shape=(latent_size,), name='Decoder-Init-H2')
    decoder_initial_state_c2 = Input(shape=(latent_size,), name='Decoder-Init-C2')
    # every layer keeps its own states, important at predicting

    i = decoder_embeddings(decoder_inputs)
    i = norm_decoder_embeddings(i)
    i, h1, c1 = decoder_lstm_1(i, initial_state=[decoder_initial_state_h1, decoder_initial_state_c1])
    lstm_inf, h2, c2 = decoder_lstm_2(i, initial_state=[decoder_initial_state_h2, decoder_initial_state_c2])
    lstm_inf = norm_decoder(lstm_inf)
    attn_out_inf, attn_states_inf = attention_layer([decoder_initial_state_a1, lstm_inf])
    decoder_concat_inf = tf.keras.layers.Concatenate(axis=-1, name='Concat')([lstm_inf, attn_out_inf])
    decoder_outputs_inf = decoder_dense(decoder_concat_inf)

    decoder_model = Model(inputs=[decoder_inputs] + [decoder_initial_state_a1,
                                                     decoder_initial_state_h1, decoder_initial_state_c1,
                                                     decoder_initial_state_h2, decoder_initial_state_c2],
                          outputs=[decoder_outputs_inf] + [h1, c1, h2, c2])

    return encoder_model, decoder_model


def greedy_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len, raw):
    encoder_out, h1, c1, h2, c2 = encoder_model.predict(input_sequence)
    # simply repeat the encoder states since  both decoding layers were trained on the encoded-vector as initialization
    target_sequence = np.array(word2idx['<START>']).reshape(1, 1)
    # populate the first character of target sequence with the start character

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, dh1, dc1, dh2, dc2 = decoder_model.predict([target_sequence] + [encoder_out, h1, c1, h2, c2])

        predicted_word_index = np.argmax(candidates)
        if predicted_word_index == 0:
            predicted_word = '<END>'
        else:
            predicted_word = idx2word[predicted_word_index]

        prediction.append(predicted_word)

        if (predicted_word == '<END>') or (len(prediction) > max_len):
            stop_condition = True

        h1, c1, h2, c2 = dh1, dc1, dh2, dc2
        target_sequence = np.array(predicted_word_index).reshape(1, 1)  # previous character

    unique = [x[0] for x in itertools.groupby(prediction[:-1])]  # remove <UNK> repetition
    final = copy_mechanism(unique, raw, word2idx)  # fill <UNK> tokens
    return ' '.join(final)


def beam_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len, raw):
    encoder_out, h1, c1, h2, c2 = encoder_model.predict(input_sequence)
    # data[i] = [[prediction], score, stop_condition, [encoder_out, h1, c1, h2, c2]] = beam

    data = [
        [[word2idx['<START>']], 0.0, False, [encoder_out, h1, c1, h2, c2]],
        [[word2idx['<START>']], 0.0, False, [encoder_out, h1, c1, h2, c2]],
        [[word2idx['<START>']], 0.0, False, [encoder_out, h1, c1, h2, c2]]]

    iteration = 1   # first iteration outside of loop
    probs, dh1, dc1, dh2, dc2 = decoder_model.predict([np.array(word2idx['<START>']).reshape(1, 1)] +
                                                      [encoder_out, h1, c1, h2, c2])
    targets = np.argpartition(probs[0][0], -3)[-3:]

    for i, target in enumerate(targets):
        beam = data[i]
        beam[0].append(target)
        beam[1] = np.log(probs[0][0][target])
        beam[3] = [encoder_out, dh1, dc1, dh2, dc2]
        data[i] = beam

    while iteration < max_len:  # predict until max sequence length reached
        iteration += 1
        candidates = []

        for i, beam in enumerate(data):
            stop_condition = beam[2]
            if not stop_condition:
                target_sequence = np.array(beam[0][-1]).reshape(1, 1)  # previous word
                states = beam[3]

                probs, dh1, dc1, dh2, dc2 = decoder_model.predict([target_sequence] + states)
                targets = np.argpartition(probs[0][0], -3)[-3:]  # predicted word indices
                score = beam[1]  # current score

                for i, target in enumerate(targets):
                    candidate = copy.deepcopy(beam)
                    candidate[0].append(target)
                    candidate[1] = score + np.log(probs[0][0][target])  # add negative values, search for max
                    candidate[3] = [encoder_out, dh1, dc1, dh2, dc2]
                    if target == 0 or target == word2idx['<END>']:
                        candidate[2] = True

                    candidates.append(candidate)  # update score, states

        candidates.sort(key=lambda x: x[1]/len(x[0]), reverse=True)  # maximize score, ascending

        next = 0
        for i in range(len(data)):
            if data[i][2] == False:
                data[i] = candidates[next]
                next += 1

    top = []
    for beam in data:
        sequence = beam[0]
        prediction = []
        for token in sequence:
            prediction.append(idx2word[token])
        top.append(prediction)

    predictions = []
    for t in top:
        unique = [x[0] for x in itertools.groupby(t[1:-1])]
        final = copy_mechanism(unique, raw, word2idx)  # fill <UNK> tokens
        predictions.append(' '.join(final))

    return predictions


def copy_mechanism(prediction, article, word2idx):
    unks = []
    for curr in article:
        if curr not in word2idx.keys():
            unks.append(curr)

    update = []
    for curr in prediction:
        if curr == '<UNK>' and len(unks) > 0:
            # directly copy first <UNK> word from article, regardless of previous word in article
            update.append(unks[0])
            unks.pop(0)
        elif curr != '<UNK>':
            update.append(curr)
        else:
            update.append('<UNK>')

    return update


def evaluate(encoder_model, decoder_model, max_len, word2idx, idx2word, titles_test, summaries_test, articles_test, raw):
    greedy_predictions = []
    beam_predictions = []

    for index in range(len(titles_test)):
        input_sequence = articles_test[index:index + 1]
        pred_greedy = greedy_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len, raw[index])
        pred_beam = beam_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len, raw[index])
        greedy_predictions.append(pred_greedy)
        beam_predictions.append(pred_beam)

        print(' '.join(titles_test[index:index + 1]))
        print(' '.join(summaries_test[index:index + 1][0]))
        print(pred_greedy)
        print(pred_beam, "\n")

    references = [' '.join(summary) for summary in summaries_test]
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=60,
                            length_limit_type='words',
                            apply_avg=False,
                            apply_best=True,
                            alpha=0.5,
                            weight_factor=1.2,
                            stemming=False)

    f = open("data/models/final_results.txt", "a", encoding="utf-8")
    f.write('\n' + 'Greedy evaluation: ' + '\n')
    greedy_scores = evaluator.get_scores(greedy_predictions, references)
    for metric, results in sorted(greedy_scores.items(), key=lambda x: x[0]):
        score = prepare_results(metric, results['p'], results['r'], results['f'])
        print('Greedy evaluation: ' + score)
        f.write('\n' + score)

    f.write('\n' + '\n' + 'Beam evaluation: ' + '\n')
    beam_scores = evaluator.get_scores(references, beam_predictions)  # reversed precision and recall
    for metric, results in sorted(beam_scores.items(), key=lambda x: x[0]):
        score = prepare_results(metric, results['p'], results['r'], results['f'])
        print('Beam evaluation: ' + score)
        f.write('\n' + score)
    f.close()


# MAIN

titles, articles, summaries = read_data()
dataset_size = len(titles)
train = int(round(dataset_size * 0.99))
test = int(round(dataset_size * 0.01))

articles = clean_data(articles, 300)
summaries = clean_data(summaries, 60)
article_min_len, article_max_len, article_avg_len = analyze_data(articles)
summary_min_len, summary_max_len, summary_avg_len = analyze_data(summaries)

embeddings_index, n, d, embedding_words = load_embeddings()
all_tokens = list(itertools.chain(*articles)) + list(itertools.chain(*summaries))
fdist, word2idx, idx2word = build_vocabulary(all_tokens, embedding_words)
vocabulary_size = len(word2idx.items())

embedding_matrix = np.zeros((vocabulary_size, 300))
embedding_matrix[1] = np.array(np.random.uniform(-1.0, 1.0, 300))  # <START>
embedding_matrix[2] = np.array(np.random.uniform(-1.0, 1.0, 300))  # <END>
embedding_matrix[3] = np.array(np.random.uniform(-1.0, 1.0, 300))  # <UNK>
for word, i in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None and i > 3:
        embedding_matrix[i] = embedding_vector

article_inputs = pre_process(articles, word2idx, True)
summary_inputs = pre_process(summaries, word2idx, False)
target_inputs = process_targets(summaries, word2idx)

article_inputs = pad_sequences(article_inputs, maxlen=article_max_len, padding='post')
summary_inputs = pad_sequences(summary_inputs, maxlen=summary_max_len, padding='post')
target_inputs = pad_sequences(target_inputs, maxlen=summary_max_len, padding='post')

article_unk, summary_unk, article_pad, summary_pad = count_padding_unknown(article_inputs, summary_inputs)

print('Dataset size (all/train/test): ', dataset_size, '/', train, '/', test)
print('Article lengths (min/max/avg): ', article_min_len, '/', article_max_len, '/', article_avg_len)
print('Summary lengths (min/max/avg): ', summary_min_len, '/', summary_max_len, '/', summary_avg_len)
print('Vocabulary size, without special tokens: ', vocabulary_size)
print('Unknown (article/summary): ', round(sum(article_unk) / len(titles), 4), '/',
      round(sum(summary_unk) / len(titles), 4))
print('Padding (article/summary): ', round(sum(article_pad) / len(titles), 4), '/',
      round(sum(summary_pad) / len(titles), 4))

train_article = article_inputs[:train]
train_summary = summary_inputs[:train]
train_target = target_inputs[:train]
test_article = article_inputs[-test:]
test_article_raw = articles[-test:]

latent_size = 512
batch_size = 16
epochs = 32

encoder_model, decoder_model = seq2seq_architecture(latent_size, vocabulary_size, article_max_len, embedding_matrix,
                                                    batch_size, epochs, train_article, train_summary, train_target)

evaluate(encoder_model, decoder_model, summary_max_len, word2idx, idx2word, titles[-test:], summaries[-test:],
         test_article, test_article_raw)
