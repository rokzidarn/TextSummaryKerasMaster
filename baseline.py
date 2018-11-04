import requests
import bs4
import nltk
import re
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Input, Model
from keras.layers import Dense, LSTM, concatenate
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

def plot_acc(history_dict, epochs):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Testing acc')
    plt.title('Training and testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def prepare_data(url):
    html = requests.get(url)
    data = bs4.BeautifulSoup(html.text, "lxml")

    title_tag = data.find(name='h1', id='firstHeading', class_='firstHeading')
    title = title_tag.text
    #print(title)

    all_text = data.find(name='div', class_='mw-parser-output')

    first_child = None
    for i in all_text.children:
        first_child = i
        break

    if first_child.name != 'p':
        tag = first_child.findNextSibling(name='p')
    else:
        tag = all_text.find('p')

    summary = [tag.text]

    while True:  # next paragraphs of summary until div tag
        tag = tag.nextSibling
        if isinstance(tag, bs4.element.Tag):
            if tag.name == 'h2':
                after = tag
                break
            elif tag.name == 'p':
                summary.append(tag.text)

    tags = after.findNextSiblings(name='p')  # all paragraphs of text
    text = []
    for t in tags:
        text.append(t.text)

    # print(summary)
    # print(text)

    summary_combined = ''
    for paragraph in summary:
        summary_combined = summary_combined + paragraph

    text_combined = ''
    for paragraph in text:
        text_combined = text_combined + paragraph

    summary_cleaned = re.sub('\[[0-9]*\]', '', summary_combined)
    text_cleaned = re.sub('\[[0-9]*\]', '', text_combined)

    # print(summary_combined)
    # print(summary_cleaned)
    # print(text_combined)
    # print(text_cleaned)

    summary_tokens = nltk.word_tokenize(summary_cleaned)
    text_tokens = nltk.word_tokenize(text_cleaned)

    exclude_list = [',', '.', '(', ')', '»', '«']
    processed_summary_tokens = [e.lower() for e in summary_tokens if e not in exclude_list]
    processed_text_tokens = [e.lower() for e in text_tokens if e not in exclude_list]

    #print(processed_summary_tokens)
    #print(processed_text_tokens)

    return (title, processed_summary_tokens, processed_text_tokens)

# MAIN

data = [
    'https://sl.wikipedia.org/wiki/Domači_pes',
    'https://sl.wikipedia.org/wiki/Nebinovke',
    'https://sl.wikipedia.org/wiki/Natrij',
    'https://sl.wikipedia.org/wiki/Kefren',
    'https://sl.wikipedia.org/wiki/Kristalna_noč'
]

titles = []
summaries = []
texts = []
all_tokens = []

for url in data:
    (title, summary, text) = prepare_data(url)
    #print(title, '\n', summary, '\n', text, '\n')

    titles.append(title)
    summaries.append(summary)
    texts.append(text)
    all_tokens = all_tokens + summary + text

freq_distribution = nltk.FreqDist(all_tokens)
vocabulary_size = len(freq_distribution.items())

encoded_summaries = [one_hot(' '.join(summary), vocabulary_size) for summary in summaries]  # summary word2index
encoded_texts = [one_hot(' '.join(text), vocabulary_size) for text in texts]  # text word2index

max_length_summary = len(max(summaries, key=len))
max_length_text = len(max(texts, key=len))

padded_summaries = pad_sequences(encoded_summaries, maxlen=max_length_summary, padding='post')
padded_texts = pad_sequences(encoded_texts, maxlen=max_length_text, padding='post')

# MODEL

"""
A second alternative model is to develop a model that generates a single word forecast and call it recursively.
That is, the decoder uses the context vector and the distributed representation of all words generated so far 
as input in order to generate the next word.

A language model can be used to interpret the sequence of words generated so far to provide a second context vector 
to combine with the representation of the source document in order to generate the next word in the sequence.

The summary is built up by recursively calling the model with the previously generated word appended 
(or, more specifically, the expected previous word during training).

The context vectors could be concentrated or added together to provide a broader context for the decoder to interpret 
and output the next word.

This is better as the decoder is given an opportunity to use the previously generated words and the source document 
as a context for generating the next word.
It does put a burden on the merge operation and decoder to interpret where it is up to in generating the output.
"""

# params
epochs = 32
embedding_size = 100

input_summary = Input(shape=(max_length_summary,))  # summary encoder
embedding_summary = Embedding(vocabulary_size, embedding_size)(input_summary)
lstm_summary = LSTM(128)(embedding_summary)

input_text = Input(shape=(max_length_text,))  # text encoder
embedding_text = Embedding(vocabulary_size, embedding_size)(input_text)
lstm_text = LSTM(128)(embedding_text)

decoder = concatenate([lstm_summary, lstm_text])
outputs = Dense(vocabulary_size, activation='softmax')(decoder)  # next word generator

model = Model(inputs=[input_summary, input_text], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#model.summary()

# source: https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/

history = model.fit([padded_summaries, padded_texts], epochs=epochs)
#history = model.fit([padded_summaries, padded_texts], [np.zeros((5, vocabulary_size), dtype=int)], epochs=epochs)

history_dict = history.history  # data during training, history_dict.keys()
gprah_epochs = range(1, epochs + 1)
plot_acc(history_dict, gprah_epochs)
