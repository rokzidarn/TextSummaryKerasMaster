import os
import nltk
import codecs
import itertools
from pprint import pprint
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding


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
    #fig.savefig('test.png')


def s2s_architecture(vocabulary_size, input_sequences, output_sequences, target_sequences):
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
    decoder_outputs, _, _ = decoder_LSTM(decoder_embeddings, initial_state=encoder_states)  # TODO: one-hot encode taget data?
    decoder_dense = Dense(vocabulary_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # training
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])
    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(x=[input_sequences, output_sequences], y=target_sequences,
                        batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # inference
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_size,))
    decoder_state_input_c = Input(shape=(latent_size,))
    decoder_input_states = [decoder_state_input_h, decoder_state_input_c]
    decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_inputs, initial_state=decoder_input_states)
    decoder_states = [decoder_h, decoder_c]
    decoder_out = decoder_dense(decoder_out)

    decoder_model = Model(inputs=[decoder_inputs] + decoder_input_states, outputs=[decoder_out] + decoder_states)

    return history, encoder_model, decoder_model


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

history, encoder_model, decoder_model = s2s_architecture(vocabulary_size, X_article, X_summary, Y_target)
