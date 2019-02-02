import os
import nltk
import codecs
import itertools
import numpy
from pprint import pprint
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras.utils import to_categorical

def read_data():
    summaries = []
    articles = []

    summary_files = os.listdir('data/summaries/')
    for file in summary_files:
        f = codecs.open('data/summaries/'+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        summaries.append(' '.join(tmp))

    article_files = os.listdir('data/articles/')
    for file in article_files:
        f = codecs.open('data/articles/'+file, encoding='utf-8')
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
    # fig.savefig('test.png')


def seq2seq_architecture(vocabulary_size, max_length_summary, input_sequences, output_sequences, target_sequences):
    # model hyper parameters
    latent_size = 128  # number of units (output dimensionality)
    embedding_size = 48  # word vector size
    batch_size = 64
    epochs = 50

    # encoder
    encoder_inputs = Input(shape=(None, ))
    encoder_embeddings = Embedding(vocabulary_size, embedding_size)(encoder_inputs)
    encoder_LSTM = LSTM(latent_size, return_state=True)
    # returns last state (hidden state + cell state), discard encoder_outputs, only keep the states
    # return state = returns the hidden state output and cell state for the last input time step
    encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_embeddings)
    encoder_states = [encoder_h, encoder_c]

    # decoder
    decoder_inputs = Input(shape=(None, ))  # set up decoder, using encoder_states as initial state
    decoder_embeddings = Embedding(vocabulary_size, embedding_size)(decoder_inputs)
    decoder_LSTM = LSTM(latent_size, return_sequences=True, return_state=True)  # return state needed for inference
    # return_sequence = returns the hidden state output for each input time step
    decoder_outputs, _, _ = decoder_LSTM(decoder_embeddings, initial_state=encoder_states)
    decoder_dense = Dense(vocabulary_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # encoded_target = one_hot_encode(target_sequences, vocabulary_size, max_length_summary)

    # training
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])
    model.summary()
    # model.save('data/model.h5')
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
    history = model.fit(x=[input_sequences, output_sequences], y=numpy.expand_dims(target_sequences, -1),
                        batch_size=batch_size, epochs=epochs)

    # inference
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_size,))
    decoder_state_input_c = Input(shape=(latent_size,))
    decoder_input_states = [decoder_state_input_h, decoder_state_input_c]
    decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_inputs, initial_state=decoder_input_states)
    decoder_states = [decoder_h, decoder_c]
    decoder_out = decoder_dense(decoder_out)

    decoder_model = Model(inputs=[decoder_inputs] + decoder_input_states, outputs=[decoder_out] + decoder_states)

    return history, model, encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, input_sequence, vocabulary_size, word2idx, idx2word, max_len):
    # encode the input as state vectors
    states_value = encoder_model.predict(input_sequence)

    # generate empty target sequence of length 1
    target_sequence = numpy.zeros((1, 1, vocabulary_size))
    # populate the first character of target sequence with the start character
    target_sequence[0, 0, word2idx['<START>']] = 1

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, h, c = decoder_model.predict(x=[target_sequence] + states_value)

        predicted_word_index = numpy.argmax(candidates)
        predicted_word = idx2word[predicted_word_index]
        prediction.append(predicted_word)

        # exit condition, either hit max length or find stop character
        if (predicted_word == '<END>') or (len(prediction) > max_len):
            stop_condition = True

        target_sequence = numpy.zeros((1, 1, vocabulary_size))
        target_sequence[0, 0, predicted_word_index] = 1

        states_value = [h, c]

    return prediction


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

# training model, encoder, decoder model needed for inference
history, model, encoder_model, decoder_model = seq2seq_architecture(vocabulary_size, max_length_summary,
                                                                    X_article, X_summary, Y_target)

# inference
for index in range(10):
    input_sequence = X_article[index:index+1]
    prediction = predict_sequence(encoder_model, decoder_model, input_sequence, vocabulary_size,
                                  word2idx, idx2word, max_length_summary)

    print('-')
    print('Input:', articles_clean[index])
    print('Output:', summaries_clean[index])
    print('Prediction:', prediction)
