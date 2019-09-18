import warnings
import os
import codecs
import re
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import nltk

def clean(text):
    tokens = nltk.word_tokenize(text)
    exclude_list = [',', '.', '(', ')', '>', '<', '»', '«', ':', '–', '-', '+', '–', '--', '/', '|', '“', '”', '•',
                    '``', '\"', "\'\'", '?', '!', ';', '*', '†', '[', ']', '%', '—', '°', '…', '=', '#', '&', '\'',
                    '$', '...', '}', '{', '„', '@', '', '//', '½', '***', '’', '·', '©']

    excluded = [e.lower() for e in tokens if e not in exclude_list]
    clean = []
    for e in excluded:
        if not any(re.findall(r'fig|pic|\+|\,|\.|å', e, re.IGNORECASE)):
            clean.append(e)

    return clean


# MAIN

datapath1 = '../data/sta/articles'
datapath2 = '../data/sta/summaries'
article_len = []
summary_len = []
i = 0
data = os.listdir(datapath1)

for file in data:
    i += 1
    if i % 10000 == 0:
        print(i)

    filepath1 = datapath1 + '/' + file
    filepath2 = (datapath2 + '/' + file).replace("src", "tgt")

    try:
        arr1 = []
        with open(filepath1, encoding='utf-8') as fp1:
            line = fp1.readline()
            arr1.append(line)
            while line:
                line = fp1.readline()
                arr1.append(line)

            article = ' '.join(arr1)

        arr2 = []
        with open(filepath2, encoding='utf-8') as fp2:
            line = fp2.readline()
            arr2.append(line)
            while line:
                line = fp2.readline()
                arr2.append(line)

            summary = ' '.join(arr2)

        #print(summary)
        #print(article)
        article_words_len = len(clean(article))
        summary_words_len = len(clean(summary))
        article_len.append(article_words_len)
        summary_len.append(summary_words_len)

    except:
        print("PROBLEM")
        continue

tmp = zip(article_len, summary_len)
filtered = list(filter(lambda x: x[0] >= 100 and x[0] <= 300 and x[1] >= 20 and x[1] <= 60, tmp))
size = len(filtered)
unzipped = [list(t) for t in zip(*filtered)]

print(size)
print(sum(unzipped[0])/size)
print(sum(unzipped[1])/size)

n, bins, patches = plt.hist(unzipped[0], 50, facecolor='blue', alpha=0.5)
print(n, bins, patches)
plt.show()

n, bins, patches = plt.hist(unzipped[1], 20, facecolor='blue', alpha=0.5)
print(n, bins, patches)
plt.show()