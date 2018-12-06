import nltk
from keras.preprocessing.sequence import pad_sequences

def build_vocabulary(summaries):
    max_length = len(max(summaries, key=len))
    words = [word for article in summaries for word in article]

    fdist = nltk.FreqDist(words)
    word2idx = {w: (i + 1) for i, (w, _) in enumerate(fdist.most_common())}  # build vocabulary
    word2idx["<PAD>"] = 0  # padding

    # print(max_length)
    # print(fdist)

    return word2idx, max_length

def preprocess(articles, word2idx):  # vectorize texts, words -> ints
    vectorized_articles = []

    for article in articles:
        article_vector = [word2idx[word] if word in word2idx else 0 for word in article]
        vectorized_articles.append(article_vector)

    return vectorized_articles

def postprocess(predictions, word2idx):  # transform ints -> words
    predicted_summaries = []
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))

    for output in predictions:
        summary_words = [idx2word[idx] if idx in idx2word else "<UNK>" for idx in output]

        predicted_summaries.append(summary_words)

    return predicted_summaries

# MAIN

summaries = [['the', 'cat', 'was', 'under', 'the', 'bed'],['the', 'cat', 'was', 'found', 'under', 'the', 'bed']]
print(summaries)

word2idx, max_length = build_vocabulary(summaries)
print(word2idx)

vectorized_summaries = preprocess(summaries, word2idx)
X_summaries = pad_sequences(vectorized_summaries, maxlen=max_length, padding='post')
print(X_summaries)  # neural network input, array of ints

Y_predictions = X_summaries
predictions = postprocess(Y_predictions, word2idx)
print(predictions)  # neural network output, predicted words
