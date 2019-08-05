import os
import nltk
import codecs
import itertools
import numpy
import matplotlib.pyplot as plt
import rouge
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, LSTM, Embedding, Dense, BatchNormalization
from tensorflow.python.keras.models import Model
# from keras.models import Model
# from keras.layers import Input, LSTM, Embedding, Dense, BatchNormalization


def read_data_train():
    summaries = []
    articles = []
    titles = []

    ddir = 'data/bert/'
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
    fig.savefig('data/models/lstm_stacked_seq2seq.png')


def seq2seq_architecture(latent_size, embedding_size, vocabulary_size, batch_size, epochs, sess):
    # encoder
    encoder_inputs = Input(shape=(None,), name='Encoder-Input')
    encoder_embeddings = Embedding(vocabulary_size, embedding_size, name='Encoder-Word-Embedding',
                                   mask_zero=False)
    norm_encoder_embeddings = BatchNormalization(name='Encoder-Batch-Normalization')

    encoder_lstm_1 = LSTM(latent_size, name='Encoder-LSTM-1', return_sequences=True, return_state=True)
    encoder_lstm_2 = LSTM(latent_size, name='Encoder-LSTM-2', return_state=True)
    # the sequence of the last layer is not returned because we want a single vector that stores everything

    e = encoder_embeddings(encoder_inputs)
    e = norm_encoder_embeddings(e)
    e, e_state_h_1, e_state_c_1 = encoder_lstm_1(e)
    e, e_state_h_2, e_state_c_2 = encoder_lstm_2(e)
    encoder_outputs = e  # the encoded, fix-sized vector which seq2seq is all about
    encoder_states = [e_state_h_2, e_state_c_2]

    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

    # decoder
    decoder_inputs = Input(shape=(None,), name='Decoder-Input')
    decoder_embeddings = Embedding(vocabulary_size, embedding_size, name='Decoder-Word-Embedding',
                                   mask_zero=False)
    norm_decoder_embeddings = BatchNormalization(name='Decoder-Batch-Normalization-1')

    decoder_lstm_1 = LSTM(latent_size, name='Decoder-LSTM-1', return_sequences=True, return_state=True)
    decoder_lstm_2 = LSTM(latent_size, name='Decoder-LSTM-2', return_sequences=True, return_state=True)
    norm_decoder = BatchNormalization(name='Decoder-Batch-Normalization-2')
    decoder_dense = Dense(vocabulary_size, activation='softmax', name="Final-Output-Dense")

    d = decoder_embeddings(decoder_inputs)
    d = norm_decoder_embeddings(d)
    d, d_state_h_1, d_state_c_1 = decoder_lstm_1(d, initial_state=encoder_states)
    d, d_state_h_2, d_state_c_2 = decoder_lstm_2(d, initial_state=encoder_states)
    d = norm_decoder(d)
    decoder_outputs = decoder_dense(d)

    seq2seq_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    # https://stackoverflow.com/questions/52465971/keras-seq2seq-stacked-layers

    seq2seq_model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy',
                          metrics=['sparse_categorical_accuracy'])

    seq2seq_model.summary()
    initialize_vars(sess)
    history = seq2seq_model.fit([X_article, X_summary], numpy.expand_dims(Y_target, -1),
                                batch_size=batch_size, epochs=epochs)

    f = open("data/models/lstm_stacked_results.txt", "w", encoding="utf-8")
    f.write("Stacked LSTM \n layers: 2 \n latent size: " + str(latent_size) + "\n embeddings size: " + str(embedding_size) + "\n")
    f.close()

    history_dict = history.history
    graph_epochs = range(1, epochs + 1)
    plot_training(history_dict, graph_epochs)

    # inference
    decoder_initial_state_h1 = Input(shape=(latent_size,), name='Decoder-Init-H1')
    decoder_initial_state_c1 = Input(shape=(latent_size,), name='Decoder-Init-C1')
    decoder_initial_state_h2 = Input(shape=(latent_size,), name='Decoder-Init-H2')
    decoder_initial_state_c2 = Input(shape=(latent_size,), name='Decoder-Init-C2')

    i = decoder_embeddings(decoder_inputs)
    i = norm_decoder_embeddings(i)
    i, h1, c1 = decoder_lstm_1(i, initial_state=[decoder_initial_state_h1, decoder_initial_state_c1])
    i, h2, c2 = decoder_lstm_2(i, initial_state=[decoder_initial_state_h2, decoder_initial_state_c2])
    i = norm_decoder(i)
    decoder_output = decoder_dense(i)
    decoder_states = [h1, c1, h2, c2]  # every layer keeps its own states, important at predicting

    decoder_model = Model(inputs=[decoder_inputs] + [decoder_initial_state_h1, decoder_initial_state_c1,
                                                     decoder_initial_state_h2, decoder_initial_state_c2],
                          outputs=[decoder_output] + decoder_states)

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len_summary):
    # encode the input as state vectors
    states_value = encoder_model.predict(input_sequence)
    # simply repeat the encoder states since  both decoding layers were trained on the encoded-vector as initialization
    states_value = states_value + states_value
    target_sequence = numpy.array(word2idx['<START>']).reshape(1, 1)
    # populate the first character of target sequence with the start character

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, h1, c1, h2, c2 = decoder_model.predict([target_sequence] + states_value)

        predicted_word_index = numpy.argmax(candidates)  # greedy search
        predicted_word = idx2word[predicted_word_index]
        prediction.append(predicted_word)

        # exit condition, either hit max length or find stop character
        if (predicted_word == '<END>') or (len(prediction) > max_len_summary):
            stop_condition = True

        states_value = [h1, c1, h2, c2]
        target_sequence = numpy.array(predicted_word_index).reshape(1, 1)  # previous character

    return prediction[:-1]


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def evaluate(encoder_model, decoder_model, titles_train, summaries_train, X_article, word2idx, idx2word, max_length_summary):
    predictions = []

    # testing
    for index in range(len(titles_train)):
        input_sequence = X_article[index]
        prediction = predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word,
                                      max_length_summary)

        predictions.append(prediction)
        print(prediction)
        f = open("data/bert/predictions/" + titles_train[index] + ".txt", "w", encoding="utf-8")
        f.write(str(prediction))
        f.close()

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

    f = open("data/models/lstm_stacked_results.txt", "a", encoding="utf-8")
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        score = prepare_results(metric, results['p'], results['r'], results['f'])
        print(score)
        f.write('\n' + score)
    f.close()


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)


# MAIN

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
encoder_model, decoder_model = seq2seq_architecture(latent_size, embedding_size, vocabulary_size, batch_size, epochs, sess)

# testing
evaluate(encoder_model, decoder_model, titles_train, summaries_train, X_article, word2idx, idx2word, max_length_summary)