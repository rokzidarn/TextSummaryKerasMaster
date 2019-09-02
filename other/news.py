import warnings
import os
import codecs
import re
warnings.simplefilter(action='ignore', category=FutureWarning)
import nltk
from bert.tokenization import FullTokenizer
import tensorflow_hub as hub
import tensorflow as tf


def create_tokenizer_from_hub_module(sess):
    bert_module = hub.Module("https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1")
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=False)


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

datapath = '../data/clarin/DNEVNIK'
data = os.listdir(datapath)

# sess = tf.Session()
# tokenizer = create_tokenizer_from_hub_module(sess)

i = 0
article_len = []
summary_len = []

for file in data:
    i += 1
    filepath = datapath + '/' + file

    try:
        with open(filepath) as fp:
            name = os.path.splitext(os.path.basename(fp.name))[0]

            arr = []
            next = True
            while next:
                line = fp.readline()
                if line.startswith('# Summary:'):
                    next = False

            summary = fp.readline()
            summary = summary.replace("è", "č").replace("È", "Č").replace('æ', 'č')
            if len(summary) < 2:
                continue
            else:
                tmp = summary.split(' ')
                tmp.pop(0)  # TODO
                summary = ' '.join(tmp)

            fp.readline()

            line = fp.readline()
            arr.append(line)
            while line:
                line = fp.readline()
                arr.append(line)

            if len(arr) > 1 and arr[0].find('(Foto:') != -1:
                arr.pop(0)

            if len(arr) > 2 and '@' in arr[-2]:
                arr.pop(-2)

            article = ''.join(arr)
            article = article.replace("è", "č").replace("È", "Č").replace('æ', 'č')

            # print(summary, '\n', article)
            # article_tokens = tokenizer.tokenize(article)
            # summary_tokens = tokenizer.tokenize(summary)
            # article_token_len = len(article_tokens)
            # summary_token_len = len(summary_tokens)

            article_words_len = len(clean(article))
            summary_words_len = len(clean(summary))
            article_len.append(article_words_len)
            summary_len.append(summary_words_len)

            print(i, article_words_len, summary_words_len)

            if article_words_len >= 150 and article_words_len <= 300 and summary_words_len >= 20 and summary_words_len <= 40:
                try:
                    with codecs.open('../data/tmp/articles/'+name+'.txt', 'w', encoding='utf8') as f:
                        f.write("{}\n".format(article))
                    with codecs.open('../data/tmp/summaries/'+name+'.txt', 'w', encoding='utf8') as f:
                        f.write("{}\n".format(summary))
                except:
                    continue
    except:
        continue


print(sum(article_len)/i)
print(sum(summary_len)/i)

tmp = zip(article_len, summary_len)
filtered = list(filter(lambda x: x[0] >= 100 and x[0] <= 300 and x[1] >= 10 and x[1] <= 60, tmp))
size = len(filtered)
print(size)

unzipped = [list(t) for t in zip(*filtered)]
print(sum(unzipped[0])/size)
print(sum(unzipped[1])/size)
