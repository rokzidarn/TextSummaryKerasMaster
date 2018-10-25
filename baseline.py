import requests
import bs4
import nltk
import re
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

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

max_length = len(max(summaries, key=len))

padded_summaries = pad_sequences(encoded_summaries, maxlen=max_length, padding='post')
padded_texts = pad_sequences(encoded_texts, maxlen=max_length, padding='post')

# MODEL


