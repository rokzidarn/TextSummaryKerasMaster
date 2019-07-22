import os
import nltk
import codecs
import itertools
import numpy
from pprint import pprint
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, TimeDistributed, BatchNormalization
from keras import optimizers
from keras.utils import to_categorical


def read_data():
    summaries = []
    articles = []

    ddir = 'data/test/'
    summary_files = os.listdir(ddir+'summaries/')
    for file in summary_files:
        f = codecs.open(ddir+'summaries/'+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        summaries.append(' '.join(tmp))

    article_files = os.listdir(ddir+'articles/')
    for file in article_files:
        f = codecs.open(ddir+'articles/'+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        articles.append(' '.join(tmp))

    return summaries, articles


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


def post_process(predictions, idx2word):  # transform array of ints (idx2word) -> array of tokens (words)
    predicted_texts = []

    for output in predictions:
        predicted_words = [idx2word[idx] if idx in idx2word else "<UNK>" for idx in output]
        predicted_texts.append(predicted_words)

    return predicted_texts


def one_hot_encode(sequences, vocabulary_size, max_length_summary):
    # to_categorical(d, num_classes=vocabulary_size)
    encoded = numpy.zeros(shape=(len(sequences), max_length_summary, vocabulary_size))
    for s in range(len(sequences)):
        for k, char in enumerate(sequences[s]):
            encoded[s, k, char] = 1

    print('Target sequences shape after one-hot encoding: ', encoded.shape)

    return encoded


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
    fig.savefig('lstm_seq2seq.png')


def predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len):
    # encode the input as state vectors
    initial_state = encoder_model.predict(input_sequence)
    initial_state = initial_state + initial_state
    # populate the first character of target sequence with the start character
    target_sequence = numpy.array(word2idx['<START>']).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, h1, c1, h2, c2 = decoder_model.predict([target_sequence] + initial_state)

        predicted_word_index = numpy.argmax(candidates)  # greedy search
        # TODO: if predicted is the same as previous, get next candidate
        predicted_word = idx2word[predicted_word_index]
        prediction.append(predicted_word)

        # exit condition, either hit max length or find stop character
        if (predicted_word == '<END>') or (len(prediction) > max_len):
            stop_condition = True

        initial_state = [h1, c1, h2, c2]
        target_sequence = numpy.array(predicted_word_index).reshape(1, 1)  # previous character

    return prediction[:-1]


# MAIN

summaries_read, articles_read = read_data()  # 1D array, each element is string of sentences, separated by newline

# 2D array, array of summaries/articles, sub-arrays of words
summaries_clean = [clean_data(summary) for summary in summaries_read]
articles_clean = [clean_data(article) for article in articles_read]

max_length_summary = len(max(summaries_clean, key=len)) + 2  # with <START> and <END> tokens added
max_length_article = len(max(articles_clean, key=len)) + 2

print('Dataset size (number of summary-article pairs): ', len(summaries_read))
print('Max lengths of summary/article in dataset: ', max_length_summary, '/', max_length_article)

all_tokens = list(itertools.chain(*summaries_clean)) + list(itertools.chain(*articles_clean))
fdist, word2idx, idx2word = build_vocabulary(all_tokens)
vocabulary_size = len(fdist.items())  # without <PAD>, <START>, <END>, <UNK> tokens

print('Vocabulary size (number of all possible words): ', vocabulary_size)
print('Most common words: ', fdist.most_common(10))
print('Vocabulary (word -> index): ', {k: word2idx[k] for k in list(word2idx)[:10]})
print('Inverted vocabulary (index -> word): ', {k: idx2word[k] for k in list(idx2word)[:10]})

# 2D array, array of summaries/articles, sub-arrays of indexes (int)
summaries_vectors = pre_process(summaries_clean, word2idx)
articles_vectors = pre_process(articles_clean, word2idx)

tmp_vectors = pre_process(summaries_clean, word2idx)  # same as summaries_vectors, but with delay
target_vectors = []  # ahead by one timestep, without start token
for tmp in tmp_vectors:
    tmp.append(word2idx['<PAD>'])  # added <PAD>, so the dimensions match
    target_vectors.append(tmp[1:])

# padded array of summaries/articles, added at the end
X_summary = pad_sequences(summaries_vectors, maxlen=max_length_summary, padding='post')
X_article = pad_sequences(articles_vectors, maxlen=max_length_article, padding='post')
Y_target = pad_sequences(target_vectors, maxlen=max_length_summary, padding='post')

# Y_encoded_target = one_hot_encode(Y_target, vocabulary_size, max_length_summary)

# model hyper parameters
latent_size = 128  # number of units (output dimensionality)
embedding_size = 96  # word vector size
batch_size = 1
epochs = 8

# training
encoder_inputs = Input(shape=(None,), name='Encoder-Input')
encoder_embeddings = Embedding(vocabulary_size, embedding_size, name='Encoder-Word-Embedding', mask_zero=False)

encoder_lstm_1 = LSTM(latent_size, name='Encoder-LSTM-1', return_sequences=True, return_state=True)
encoder_lstm_2 = LSTM(latent_size, name='Encoder-LSTM-2', return_state=True)
# the sequence of the last layer is not returned because we want a single vector that stores everything

e = encoder_embeddings(encoder_inputs)
e, _, _ = encoder_lstm_1(e)
e, e_state_h, e_state_c = encoder_lstm_2(e)
encoder_outputs = e  # the encoded, fix-sized vector which seq2seq is all about
encoder_states = [e_state_h, e_state_c]

decoder_initial_state_h1 = Input(shape=(latent_size,), name='Decoder-Init-H1')
decoder_initial_state_c1 = Input(shape=(latent_size,), name='Decoder-Init-C1')
decoder_initial_state_h2 = Input(shape=(latent_size,), name='Decoder-Init-H2')
decoder_initial_state_c2 = Input(shape=(latent_size,), name='Decoder-Init-C2')

decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # set up decoder, using encoder_states as initial state
decoder_embeddings = Embedding(vocabulary_size, embedding_size, name='Decoder-Word-Embedding', mask_zero=False)

decoder_lstm_1 = LSTM(latent_size, name='Decoder-LSTM-1', return_sequences=True, return_state=True)
decoder_lstm_2 = LSTM(latent_size, name='Decoder-LSTM-2', return_sequences=True, return_state=True)
decoder_dense = Dense(vocabulary_size, activation='softmax', name="Final-Output-Dense")

# feed the encoder_states as initial input to both decoding lstm layers
d = decoder_embeddings(decoder_inputs)
d, d_state_h, d_state_c = decoder_lstm_1(d, initial_state=encoder_states)
d, _, _ = decoder_lstm_2(d, initial_state=encoder_states)  # TODO
decoder_outputs = decoder_dense(d)

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
seq2seq_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

seq2seq_model.summary()
seq2seq_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

history = seq2seq_model.fit([X_article, X_summary], numpy.expand_dims(Y_target, -1), batch_size=batch_size, epochs=epochs)

# inference

i = decoder_embeddings(decoder_inputs)
i, h1, c1 = decoder_lstm_1(i, initial_state=[decoder_initial_state_h1, decoder_initial_state_c1])
i, h2, c2 = decoder_lstm_2(i, initial_state=[decoder_initial_state_h2, decoder_initial_state_c2])
decoder_output = decoder_dense(i)
decoder_states = [h1, c1, h2, c2]  # every layer keeps its own states, important at predicting

decoder_model = Model(inputs=[decoder_inputs] + [decoder_initial_state_h1, decoder_initial_state_c1,
                                                 decoder_initial_state_h2, decoder_initial_state_c2],
                      outputs=[decoder_output] + decoder_states)

# testing
for index in range(5):
    input_sequence = X_article[index:index+1]
    prediction = predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_length_summary)

    print('-')
    print('Article:', articles_clean[index])
    print('Summary:', summaries_clean[index])
    print('Prediction:', prediction)
