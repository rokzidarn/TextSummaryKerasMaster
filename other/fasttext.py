from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import io
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import tensorflow as tf


def gpu_test():
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        print(sess.run(c))


def class_weights():
    y_train = [[0, 1, 1, 3, 2, 4], [1, 1, 2, 4, 0, 0]]
    y_train = [item for sublist in y_train for item in sublist]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print(class_weights)

    # targets = [item for sublist in target_input_ids for item in sublist]
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(targets), targets)


def load_embeddings():
    fin = io.open('../data/fasttext/wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
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


# MAIN
# https://github.com/PacktPublishing/fastText-Quick-Start-Guide/blob/master/chapter6/keras%20fasttext%20convnets.ipynb

data = ['I love machine learning',
        'I don\'t like reading books.',
        'Python Jože is horrible',
        'Machine learning is cool!',
        'I really like NLP']

labels = ['positive', 'negative', 'negative', 'positive', 'positive']

text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in data]
text = [sentence.lower().split() for sentence in text]
words = set([item for sublist in text for item in sublist])
print(text)
print(len(words), words)

tokenizer = Tokenizer(num_words=len(words))
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(text)
X = pad_sequences(X, maxlen=len(max(text, key=len)), padding='post')
print(word_index)
print(X)

y = pd.get_dummies(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

embeddings_index, n, d, words = load_embeddings()
print(n,d,words[:100]);exit()

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        # embedding_matrix[i] = np.array(np.random.uniform(-1.0, 1.0, 300))
        print(word)

model = Sequential()
model.add(Embedding(len(word_index)+1, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
model.add(LSTM(300, return_sequences=False))
model.add(Dense(y.shape[1], activation="softmax"))
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])

batch = 32
epochs = 12
model.fit(X_train, y_train, batch, epochs)
model.evaluate(X_test, y_test)
