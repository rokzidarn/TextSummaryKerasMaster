import os
import nltk
import codecs
import itertools
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import rouge
from attention import AttentionLayer


def read_data():
    summaries = []
    articles = []
    titles = []

    ddir = 'data/test/'
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
    fig.savefig('data/models/gru_seq2seq.png')


def seq2seq_architecture(vocabulary_size, max_len_article, max_len_summary):
    # model hyper parameters
    latent_size = 128  # number of units (output dimensionality)
    embedding_size = 96  # word vector size
    batch_size = 8
    epochs = 12

    f = open("data/models/gru_results.txt", "w")
    f.write(
        "GRU \n layers: 1 \n latent size: " + str(latent_size) + "\n embeddings size: " + str(embedding_size) + "\n")
    f.close()

    encoder_inputs = tf.keras.layers.Input(shape=(max_len_article,), name='Encoder-Input')
    encoder_embeddings = tf.keras.layers.Embedding(vocabulary_size, embedding_size, name='Encoder-Word-Embedding',
                                                   mask_zero=False)(encoder_inputs)
    encoder_embeddings = tf.keras.layers.BatchNormalization(name='Encoder-Batch-Normalization')(encoder_embeddings)
    encoder_outputs, state_h = tf.keras.layers.GRU(latent_size, return_state=True, return_sequences=True,
                                                   name='Encoder-GRU')(encoder_embeddings)

    decoder_inputs = tf.keras.layers.Input(shape=(None,), name='Decoder-Input')
    decoder_embeddings_layer = tf.keras.layers.Embedding(vocabulary_size, embedding_size, name='Decoder-Word-Embedding',
                                                         mask_zero=False)
    decoder_embeddings = decoder_embeddings_layer(decoder_inputs)

    decoder_batchnorm_layer = tf.keras.layers.BatchNormalization(name='Decoder-Batch-Normalization-1')
    decoder_batchnormalization = decoder_batchnorm_layer(decoder_embeddings)
    decoder_gru = tf.keras.layers.GRU(latent_size, return_state=True, return_sequences=True, name='Decoder-GRU')
    decoder_gru_outputs, _ = decoder_gru(decoder_batchnormalization, initial_state=state_h)

    attention_layer = AttentionLayer(name='Attention')
    attn_out, attn_states = attention_layer([encoder_outputs, decoder_gru_outputs])
    decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='Concat')([decoder_gru_outputs, attn_out])

    decoder_dense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(vocabulary_size, activation='softmax', name='Final-Output-Dense'), name='Time-Distributed')
    decoder_outputs = decoder_dense(decoder_concat_input)

    seq2seq_model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    seq2seq_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy',
                          metrics=['sparse_categorical_accuracy'])

    seq2seq_model.summary()

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)  # stop training once validation loss increases
    history = seq2seq_model.fit([X_article, X_summary],
                                numpy.expand_dims(Y_target, -1), batch_size=batch_size, epochs=epochs)

    history_dict = history.history
    graph_epochs = range(1, epochs + 1)
    plot_training(history_dict, graph_epochs)

    encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h], name='Encoder-Model')

    decoder_state_input_h = tf.keras.layers.Input(shape=(latent_size,))
    decoder_hidden_state_input = tf.keras.layers.Input(shape=(max_len_article, latent_size))

    decoder_embeddings_inf = decoder_embeddings_layer(decoder_inputs)
    decoder_batchnorm_inf = decoder_batchnorm_layer(decoder_embeddings_inf)
    decoder_outputs_inf, state_h_inf = decoder_gru(decoder_batchnorm_inf, initial_state=[decoder_state_input_h])

    attn_out_inf, attn_states_inf = attention_layer([decoder_hidden_state_input, decoder_outputs_inf])
    decoder_inf_concat = tf.keras.layers.Concatenate(axis=-1, name='Concat-Inference')([decoder_outputs_inf, attn_out_inf])
    decoder_outputs_inf = decoder_dense(decoder_inf_concat)

    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h], [decoder_outputs_inf] + [state_h_inf])

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len):
    encoder_out, states_value = encoder_model.predict(input_sequence)
    target_sequence = numpy.array(word2idx['<START>']).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, state = decoder_model.predict([target_sequence] + [encoder_out, states_value])

        predicted_word_index = numpy.argmax(candidates)  # greedy search
        predicted_word = idx2word[predicted_word_index]
        prediction.append(predicted_word)

        if (predicted_word == '<END>') or (len(prediction) > max_len):
            stop_condition = True

        states_value = state
        target_sequence = numpy.array(predicted_word_index).reshape(1, 1)  # previous character

    return prediction[:-1]


def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


# MAIN

# https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/

titles, summaries_read, articles_read = read_data()
dataset_size = len(titles)

summaries_clean = [clean_data(summary) for summary in summaries_read]
articles_clean = [clean_data(article) for article in articles_read]

max_length_summary = len(max(summaries_clean, key=len)) + 2  # with <START> and <END> tokens added
max_length_article = len(max(articles_clean, key=len)) + 2

print('Dataset size (number of summary-article pairs): ', len(summaries_read))
print('Max lengths of summary/article in dataset: ', max_length_summary, '/', max_length_article)

all_tokens = list(itertools.chain(*summaries_clean)) + list(itertools.chain(*articles_clean))
fdist, word2idx, idx2word = build_vocabulary(all_tokens)
vocabulary_size = len(word2idx.items())  # with <PAD>, <START>, <END>, <UNK> tokens

print('Vocabulary size (number of all possible words): ', vocabulary_size)
print('Most common words: ', fdist.most_common(10))
print('Vocabulary (word -> index): ', {k: word2idx[k] for k in list(word2idx)[:10]})
print('Inverted vocabulary (index -> word): ', {k: idx2word[k] for k in list(idx2word)[:10]})

summaries_vectors = pre_process(summaries_clean, word2idx)
articles_vectors = pre_process(articles_clean, word2idx)

tmp_vectors = pre_process(summaries_clean, word2idx)  # same as summaries_vectors, but with delay
target_vectors = []  # ahead by one timestep, without start token
for tmp in tmp_vectors:
    tmp.append(word2idx['<PAD>'])  # added <PAD>, so the dimensions match
    target_vectors.append(tmp[1:])

X_summary = tf.keras.preprocessing.sequence.pad_sequences(summaries_vectors, maxlen=max_length_summary, padding='post')
X_article = tf.keras.preprocessing.sequence.pad_sequences(articles_vectors, maxlen=max_length_article, padding='post')
Y_target = tf.keras.preprocessing.sequence.pad_sequences(target_vectors, maxlen=max_length_summary, padding='post')

encoder_model, decoder_model = seq2seq_architecture(vocabulary_size, max_length_article, max_length_summary)

predictions = []

# testing
for index in range(5):
    input_sequence = X_article[index:index+1]
    prediction = predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_length_summary)
    predictions.append(prediction)

    print('-')
    print('Summary:', summaries_clean[index])
    print('Prediction:', prediction)

# evaluation using ROUGE
aggregator = 'Best'
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
all_references = [' '.join(summary) for summary in summaries_clean[:5]]
scores = evaluator.get_scores(all_hypothesis, all_references)

f = open("data/models/gru_results.txt", "w")
print('\n ROUGE evaluation: ')
for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print('\n', prepare_results(results['p'], results['r'], results['f']))
    f.write('\n' + prepare_results(results['p'], results['r'], results['f']))
f.close()
