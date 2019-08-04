import os
import nltk
import codecs
import itertools
import numpy
import matplotlib.pyplot as plt
import rouge
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, GRU, Embedding, Dense, BatchNormalization, Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.python.keras.models import Model
# from keras.models import Model
# from keras.layers import Input, GRU, Dense, Embedding, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Dropout


def read_data_train():
    summaries = []
    articles = []
    titles = []

    ddir = 'data/bert/'
    summary_files = os.listdir(ddir+'summaries/')
    for file in summary_files:
        f = codecs.open(ddir+'summaries/'+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        summaries.append(' '.join(tmp))
        titles.append(file[:-4])

    article_files = os.listdir(ddir+'articles/')
    for file in article_files:
        f = codecs.open(ddir+'articles/'+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        articles.append(' '.join(tmp))

    return titles, summaries, articles


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


def process_targets(summaries_train, word2idx):
    tmp_vectors = pre_process(summaries_train, word2idx)  # same as summaries_vectors, but with delay
    target_vectors = []  # ahead by one timestep, without start token
    for tmp in tmp_vectors:
        tmp.append(word2idx['<PAD>'])  # added <PAD>, so the dimensions match
        target_vectors.append(tmp[1:])

    return target_vectors


def post_process(predictions, idx2word):  # transform array of ints (idx2word) -> array of tokens (words)
    predicted_texts = []

    for output in predictions:
        predicted_words = [idx2word[idx] if idx in idx2word else "<UNK>" for idx in output]
        predicted_texts.append(predicted_words)

    return predicted_texts


def plot_training(history_dict, epochs):
    loss = history_dict['loss']

    fig = plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    fig.savefig('data/models/conv_seq2seq.png')


def seq2seq_architecture(max_length_article, latent_size, embedding_size, vocabulary_size):
    # encoder
    encoder_inputs = Input(shape=(max_length_article,), name='Encoder-Input')
    encoder_embeddings = Embedding(vocabulary_size, embedding_size, name='Encoder-Word-Embedding',
                                   mask_zero=False)(encoder_inputs)
    encoder_embeddings = BatchNormalization(name='Encoder-Batch-Normalization')(encoder_embeddings)
    encoder_conv = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(encoder_embeddings)
    encoder_drop = Dropout(0.25)(encoder_conv)
    encoder_pool = MaxPooling1D(pool_size=4)(encoder_drop)
    encoder_flatten = Flatten()(encoder_pool)
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_flatten, name='Encoder-Model')
    encoder_outputs = encoder_model(encoder_inputs)

    # decoder
    decoder_inputs = Input(shape=(None,), name='Decoder-Input')
    decoder_embeddings = Embedding(vocabulary_size, embedding_size, name='Decoder-Word-Embedding',
                                   mask_zero=False)(decoder_inputs)
    decoder_embeddings = BatchNormalization(name='Decoder-Batchnormalization-1')(decoder_embeddings)
    decoder_conv = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu', name='Decoder-Conv1D') \
        (decoder_embeddings)
    decoder_drop = Dropout(0.25, name='Decoder-Conv1D-Dropout')(decoder_conv)
    decoder_pool = MaxPooling1D(pool_size=1, name='Decoder-MaxPool1D')(decoder_drop)  # GlobalMaxPool1D()

    decoder_gru = GRU(latent_size, return_state=True, return_sequences=True, name='Decoder-GRU')
    decoder_gru_outputs, _ = decoder_gru(decoder_pool, initial_state=encoder_outputs)
    decoder_outputs = BatchNormalization(name='Decoder-Batchnormalization-2')(decoder_gru_outputs)
    decoder_outputs = Dense(vocabulary_size, activation='softmax', name='Final-Output-Dense')(decoder_outputs)

    seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return seq2seq_model


def inference(model):
    encoder_model = model.get_layer('Encoder-Model')

    latent_dim = model.get_layer('Decoder-Word-Embedding').output_shape[-1]  # gets embedding size, not latent size
    decoder_inputs = model.get_layer('Decoder-Input').input
    decoder_embeddings = model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
    decoder_embeddings = model.get_layer('Decoder-Batchnormalization-1')(decoder_embeddings)
    decoder_conv = model.get_layer('Decoder-Conv1D')(decoder_embeddings)
    decoder_drop = model.get_layer('Decoder-Conv1D-Dropout')(decoder_conv)
    decoder_pool = model.get_layer('Decoder-MaxPool1D')(decoder_drop)
    gru_inference_state_input = Input(shape=(latent_dim,), name='Hidden-State-Input')
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([decoder_pool, gru_inference_state_input])
    decoder_outputs = model.get_layer('Decoder-Batchnormalization-2')(gru_out)
    dense_out = model.get_layer('Final-Output-Dense')(decoder_outputs)
    decoder_model = Model([decoder_inputs, gru_inference_state_input], [dense_out, gru_state_out])

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len_summary):
    # encode the input as state vectors
    states_value = encoder_model.predict(input_sequence)
    # populate the first character of target sequence with the start character
    target_sequence = numpy.array(word2idx['<START>']).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, state = decoder_model.predict([target_sequence, states_value])

        predicted_word_index = numpy.argmax(candidates)  # greedy search
        predicted_word = idx2word[predicted_word_index]
        prediction.append(predicted_word)

        # exit condition, either hit max length or find stop character
        if (predicted_word == '<END>') or (len(prediction) > max_len_summary):
            stop_condition = True

        states_value = state
        target_sequence = numpy.array(predicted_word_index).reshape(1, 1)  # previous character

    return prediction[:-1]


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def evaluate(encoder_model, decoder_model, titles_train, summaries_train, X_article, word2idx, idx2word, max_length_summary):
    predictions = []

    # testing
    for index in range(len(titles_train)):
        input_sequence = X_article[index:index+1]
        prediction = predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word,
                                      max_length_summary)

        predictions.append(prediction)
        print(prediction)
        f = open("data/bert/predictions/" + titles_train[index] + ".txt", "w", encoding="utf-8")
        f.write(str(prediction))
        f.close()

    # evaluation using ROUGE
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=3,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=False,
                            apply_best=True,
                            alpha=0.5,  # default F1 score
                            weight_factor=1.2,
                            stemming=True)

    all_hypothesis = [' '.join(prediction) for prediction in predictions]
    all_references = [' '.join(summary) for summary in summaries_train]
    scores = evaluator.get_scores(all_hypothesis, all_references)

    f = open("data/models/conv_results.txt", "a", encoding="utf-8")
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        score = prepare_results(metric, results['p'], results['r'], results['f'])
        print(score)
        f.write('\n' + score)
    f.close()


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)


# MAIN

sess = tf.Session()

# 1D array, each element is string of sentences, separated by newline
titles, summaries_read, articles_read = read_data_train()

# 2D array, array of summaries/articles, sub-arrays of words
summaries_clean = [clean_data(summary) for summary in summaries_read]
articles_clean = [clean_data(article) for article in articles_read]

max_length_summary = len(max(summaries_clean, key=len)) + 2  # with <START> and <END> tokens added
max_length_article = len(max(articles_clean, key=len)) + 2

all_tokens = list(itertools.chain(*summaries_clean)) + list(itertools.chain(*articles_clean))
fdist, word2idx, idx2word = build_vocabulary(all_tokens)
vocabulary_size = len(word2idx.items())  # with <PAD>, <START>, <END>, <UNK> tokens

print('Dataset size (number of summary-article pairs): ', len(summaries_read))
print('Max lengths of summary/article in dataset: ', max_length_summary, '/', max_length_article)
print('Vocabulary size (number of all possible words): ', vocabulary_size)
print('Most common words: ', fdist.most_common(10))
print('Vocabulary (word -> index): ', {k: word2idx[k] for k in list(word2idx)[:10]})
print('Inverted vocabulary (index -> word): ', {k: idx2word[k] for k in list(idx2word)[:10]})

# 2D array, array of summaries/articles, sub-arrays of indexes (int)
summaries_vectors = pre_process(summaries_clean, word2idx)
articles_vectors = pre_process(articles_clean, word2idx)
target_vectors = process_targets(summaries_clean, word2idx)

# padded array of summaries/articles, added at the end
X_summary = pad_sequences(summaries_vectors, maxlen=max_length_summary, padding='post')
X_article = pad_sequences(articles_vectors, maxlen=max_length_article, padding='post')
Y_target = pad_sequences(target_vectors, maxlen=max_length_summary, padding='post')

# model hyper parameters
latent_size = 96  # number of units (output dimensionality)
embedding_size = 96  # word vector size
batch_size = 1
epochs = 12

# training
seq2seq_model = seq2seq_architecture(max_length_article, latent_size, embedding_size, vocabulary_size)
seq2seq_model.summary()
seq2seq_model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
initialize_vars(sess)
history = seq2seq_model.fit(x=[X_article, X_summary], y=numpy.expand_dims(Y_target, -1),
                            batch_size=batch_size, epochs=epochs)

f = open("data/models/conv_results.txt", "w", encoding="utf-8")
f.write("Conv \n layers: 1 \n latent size: " + str(latent_size) + "\n embeddings size: " + str(embedding_size) + "\n")
f.close()

history_dict = history.history
graph_epochs = range(1, epochs + 1)
plot_training(history_dict, graph_epochs)

# inference
encoder_model, decoder_model = inference(seq2seq_model)

# evaluation using ROUGE
evaluate(encoder_model, decoder_model, titles, summaries_clean, X_article, word2idx, idx2word, max_length_summary)
