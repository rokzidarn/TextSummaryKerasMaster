import io
import numpy as np
from itertools import groupby
from nltk import FreqDist
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings():
    fin = io.open('../data/fasttext/wiki.sl.align.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    embeddings_index = {}
    words = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        words.append(word)
        coefs = np.asarray(tokens[1:], dtype='float32')
        embeddings_index[word] = coefs
    return embeddings_index, words


def copy_mechanism(article, summary, unks):
    tokens = (article + ' ' + summary).split(' ')
    all = FreqDist(tokens).most_common()

    vocab = {w: i for i, (w, _) in enumerate(all)}
    for u in unks:
        del vocab[u]

    real_article = article.split(' ')
    real_summary = summary.split()
    prediction_summary = [w if w in vocab.keys() else '<UNK>' for w in real_summary]
    unks_article = []
    for i in range(1, len(real_article)):
        curr = real_article[i]
        prev = real_article[i-1]
        if curr not in vocab.keys():
            unks_article.append((prev, curr))

    #print(real_summary, '\n', prediction_summary, '\n', unks_article)

    pointer = []
    for i in range(len(prediction_summary)):
        curr = prediction_summary[i]
        if curr == '<UNK>' and len(unks_article) > 0:
            # directly copy first <UNK> word from article, regardless of previous word in article
            pointer.append(unks_article[0][1])
            unks_article.pop(0)
        elif curr != '<UNK>':
            pointer.append(curr)
        else:
            pointer.append('<UNK>')

    print(pointer)


def copy_mechanism_with_similarity(article, summary, unks):
    embeddings_index, embedding_words = load_embeddings()
    tokens = (article + ' ' + summary).split(' ')
    all = FreqDist(tokens).most_common()

    vocab = {w: i for i, (w, _) in enumerate(all)}
    for u in unks:
        del vocab[u]

    real_article = article.split(' ')
    real_summary = summary.split()
    prediction_summary = [w if w in vocab.keys() else '<UNK>' for w in real_summary]
    unks_article = []
    for i in range(1, len(real_article)):
        curr = real_article[i]
        prev = real_article[i-1]
        if curr not in vocab.keys():
            unks_article.append((prev, curr))

    #print(real_summary, '\n', prediction_summary, '\n', unks_article)

    pointer = []
    for i in range(len(prediction_summary)):
        curr = prediction_summary[i]
        if curr == '<UNK>' and i == 0 and len(unks_article) > 0:
            # directly copy first <UNK> word from article, regardless of previous word in article, because can't compare
            pointer.append(unks_article[0][1])
            unks_article.pop(0)
        elif curr == '<UNK>' and i != 0 and len(unks_article) > 0:
            # cosine similarity, find best, compare previous word in article and prediction, no consecutive <UNK>
            if unks_article[0][0] == '<START>':
                pointer.append(unks_article[0][1])  # directly copy first <UNK> word from article
                unks_article.pop(0)
            else:
                prev = prediction_summary[i-1]
                candidates = []
                for j in range(len(unks_article)):
                    (p, c) = unks_article[j]
                    if p in embedding_words and prev in embedding_words:
                        sim = cosine_similarity(np.array([embeddings_index[prev]]), np.array([embeddings_index[p]]))
                        candidates.append((j, c, sim))
                    else:
                        candidates.append((j, c, [0.0]))
                (idx, best, score) = max(candidates, key=lambda item: item[2][0])
                pointer.append(best)
                unks_article.pop(idx)

        elif curr != '<UNK>':
            pointer.append(curr)
        else:
            pointer.append('<UNK>')

    print(pointer)


# MAIN

article = '<START> gospod bavčar je dosegel sporazum s katerim bo njegov dolg poplačan s strani nlb ' \
          's čimer je to postala težava trenutne vlade vendar tega premier ne želi priznati kar buri javnost <END>'

summary1 = 'dolg bavčar bo plačala nlb javnost razburjena'
unks1 = ['bavčar', 'nlb', 'težava']  # pravilen vrsti red, nepopoln povzetek

summary2 = 'dolgove bo gospod bavčar plačal slovenska javnost navdušena'
unks2 = ['dolgove', 'bavčar', 's']  # nepravilen vrsti red, nepopoln povzetek

summary3 = 'državljani bomo plačali dolg bavčarja'
unks3 = ['državljani', 'bavčarja', 'javnost']  # pravilen vrsti red, nepopoln članek

copy_mechanism(article, summary1, unks1)
copy_mechanism_with_similarity(article, summary1, unks1)

exit()
# ----------------------------------------------------------------------------------------------------------------------

stop_condition = False
prediction = ['<UNK>', '<UNK>']
# prediction = ['rekel', '<UNK>']
# prediction = ['<UNK>', 'je']
predicted_word = '<UNK>'
if predicted_word == '<UNK>' and len(prediction) > 1:
    if prediction[-1] == '<UNK>' and prediction[-2] == '<UNK>':
        stop_condition = True  # stop predicting
    else:
        prediction.append('<UNK>')  # continue predicting

print(prediction)
# ----------------------------------------------------------------------------------------------------------------------

L1 = ['<UNK>', '<UNK>', '<UNK>', '<UNK>', '<UNK>', 'je', 'rekel', 'Matej' '<UNK>', 'da']
L2 = ['Mi', '<UNK>', 'junaki', '<UNK>', '<UNK>', 'vse', 'kar', 'je', 'v']
L3 = ['Ti', 'pa', '<UNK>', '<UNK>', '<UNK>', '<UNK>', 'ne', 'ne']
L4 = ['<UNK>', 'tudi', 'tudi', 'mi', '<UNK>', '<UNK>', '<UNK>', '<UNK>', 'da', 'smo', '<UNK>', '<UNK>', '<UNK>']

print([x[0] for x in groupby(L1)])
