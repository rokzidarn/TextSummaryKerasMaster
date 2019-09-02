import os
import codecs
import itertools
import re
import nltk
import matplotlib.pyplot as plt
import numpy
import rouge
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.python.keras.layers import Input, LSTM, Embedding, Dense, BatchNormalization
from tensorflow.python.keras.models import Model
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense, Embedding, BatchNormalization


def read_data():
    summaries = []
    articles = []
    titles = []

    ddir = 'data/news/'

    article_files = os.listdir(ddir + 'articles/')
    for file in article_files:
        f = codecs.open(ddir + 'articles/' + file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        articles.append(' '.join(tmp))

    summary_files = os.listdir(ddir+'summaries/')
    for file in summary_files:
        f = codecs.open(ddir+'summaries/'+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        summaries.append(' '.join(tmp))
        titles.append(file[:-4])

    return titles, articles, summaries


def plot_loss(history_dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = list(range(1, len(loss)+1))

    fig = plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('data/models/ft_seq2seq_train.png')

    fig = plt.figure()
    plt.plot(epochs, val_loss, 'r')
    plt.title('Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('data/models/ft_seq2seq_valid.png')


def clean_data(data):
    cleaned = []

    for text in data:
        tokens = nltk.word_tokenize(text)
        exclude_list = [',', '.', '(', ')', '>', '<', '»', '«', ':', '–', '-', '+', '–', '--', '/', '|', '“', '”', '•',
                        '``', '\"', "\'\'", '?', '!', ';', '*', '†', '[', ']', '%', '—', '°', '…', '=', '#', '&', '\'',
                        '$', '...', '}', '{', '„', '@', '', '//', '½', '***', '’', '·', '©']

        # keeps dates and numbers, excludes the rest
        excluded = [e.lower() for e in tokens if e not in exclude_list]

        # remove decimals, html texts
        clean = []
        for e in excluded:
            if not any(re.findall(r'fig|pic|\+|\,|\.|å', e, re.IGNORECASE)):
                clean.append(e)

        cleaned.append(clean)

    return cleaned


def analyze_data(data, show_plot=False):
    lengths = [len(text) for text in data]
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = int(round(sum(lengths)/len(lengths)))

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


def build_vocabulary(tokens, write_dict=False):
    fdist = nltk.FreqDist(tokens)
    # fdist.pprint(maxlen=50)
    # fdist.plot(50)

    all = fdist.most_common()
    unique_words = fdist.hapaxes()
    sub_all = [element for element in all if element[1] > 20]

    word2idx = {w: (i + 4) for i, (w, _) in enumerate(all)}  # vocabulary, 'word' -> 11
    # word2idx['<PAD>'] = 0  # padding
    word2idx['<START>'] = 1  # start token
    word2idx['<END>'] = 2  # end token
    word2idx['<UNK>'] = 3  # unknown token
    # with <START>, <END>, <UNK> tokens, without <PAD> token

    idx2word = {v: k for k, v in word2idx.items()}  # inverted vocabulary, for decoding, 11 -> 'word'

    if write_dict:
        f = open("data/models/data_dict.txt", "w", encoding='utf-8')
        for k, v in word2idx.items():
            f.write(k+'\n')
        f.close()

    return fdist, unique_words, word2idx, idx2word


def pre_process(texts, word2idx):
    vectorized_texts = []

    for text in texts:  # vectorizes texts, array of tokens (words) -> array of ints (word2idx)
        text_vector = [word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in text]
        text_vector.insert(0, word2idx['<START>'])  # add <START> and <END> tokens to each summary/article
        text_vector.append(word2idx['<END>'])
        vectorized_texts.append(text_vector)

    return vectorized_texts


def process_targets(summaries, word2idx):
    tmp_inputs = pre_process(summaries, word2idx)  # same as summaries_vectors, but with delay
    target_inputs = []  # ahead by one timestep, without start token
    for tmp in tmp_inputs:
        tmp.append(0)  # added 0 for padding, so the dimensions match
        target_inputs.append(tmp[1:])

    return target_inputs


# MAIN

titles, articles, summaries = read_data()
dataset_size = len(titles)
train = int(round(dataset_size * 0.9))
test = int(round(dataset_size * 0.1))

articles = clean_data(articles)
summaries = clean_data(summaries)
article_min_len, article_max_len, article_avg_len = analyze_data(articles)
summary_min_len, summary_max_len, summary_avg_len = analyze_data(summaries)

all_tokens = list(itertools.chain(*articles)) + list(itertools.chain(*summaries))
fdist, unique_words, word2idx, idx2word = build_vocabulary(all_tokens)
vocabulary_size = len(word2idx.items())

print('Dataset size (all/train/test): ', dataset_size, '/', train, '/', test)
print('Article lengths (min/max/avg): ', article_min_len, '/', article_max_len, '/', article_avg_len)
print('Summary lengths (min/max/avg): ', summary_min_len, '/', summary_max_len, '/', summary_avg_len)
print('Vocabulary size, unique words: ', vocabulary_size, '/', len(unique_words))
print('Inverted vocabulary: ', {k: idx2word[k] for k in list(idx2word)[:15]})

article_inputs = pre_process(articles, word2idx)
summary_inputs = pre_process(summaries, word2idx)
target_inputs = process_targets(summaries, word2idx)

article_inputs = pad_sequences(article_inputs, maxlen=article_max_len, padding='post')
summary_inputs = pad_sequences(summary_inputs, maxlen=summary_max_len, padding='post')
target_inputs = pad_sequences(target_inputs, maxlen=summary_max_len, padding='post')

train_article = article_inputs[:train]
train_summary = summary_inputs[:train]
train_target = target_inputs[:train]
test_article = article_inputs[-test:]


