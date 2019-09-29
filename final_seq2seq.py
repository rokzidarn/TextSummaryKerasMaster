import os
import io
import codecs
import itertools
import re
import nltk
import rouge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Input, LSTM, Embedding, Dense, BatchNormalization
from tensorflow.python.keras.models import Model
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense, Embedding, BatchNormalization


def read_data():
    summaries = []
    articles = []
    titles = []

    ddir = 'data/test/'

    article_files = os.listdir(ddir + 'articles/')
    for file in article_files:
        f = codecs.open(ddir + 'articles/' + file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        articles.append(' '.join(tmp))

    summary_files = os.listdir(ddir + 'summaries/')
    for file in summary_files:
        f = codecs.open(ddir + 'summaries/' + file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        summaries.append(' '.join(tmp))
        titles.append(file[:-4])

    return titles, articles, summaries


def load_embeddings():
    fin = io.open('data/fasttext/wiki.sl.align.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
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


def plot_loss(history_dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = list(range(1, len(loss) + 1))

    fig = plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('data/models/final_train.png')

    fig = plt.figure()
    plt.plot(epochs, val_loss, 'r')
    plt.title('Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('data/models/final_valid.png')


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1',
                                                                 100.0 * f)


def clean_data(data):
    cleaned = []

    for text in data:
        tokens = nltk.word_tokenize(text)
        exclude_list = [',', '.', '(', ')', '>', '<', '»', '«', ':', '–', '-', '+', '–', '--', '/', '|', '“', '”', '•',
                        '``', '\"', "\'\'", '?', '!', ';', '*', '†', '[', ']', '%', '—', '°', '…', '=', '#', '&', '\'',
                        '$', '...', '}', '{', '„', '@', '', '//', '½', '***', '’', '·', '©']

        # keeps dates and numbers, excludes the rest
        excluded = [e for e in tokens if e not in exclude_list]  # lower()

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
    avg_len = int(round(sum(lengths) / len(lengths)))

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


def build_vocabulary(tokens, embedding_words, write_dict=False):
    fdist = nltk.FreqDist(tokens)
    # fdist.pprint(maxlen=50)
    # fdist.plot(50)

    all = fdist.most_common()  # unique_words = fdist.hapaxes()
    sub_all = [element for element in all if element[1] > 25]  # cut vocabulary

    embedded = []  # exclude words that are not in embedding matrix
    for element in sub_all:
        w, r = element
        if w in embedding_words:
            embedded.append(element)

    word2idx = {w: (i + 4) for i, (w, _) in enumerate(embedded)}
    # word2idx['<PAD>'] = 0  # padding
    word2idx['<START>'] = 1  # start token
    word2idx['<END>'] = 2  # end token
    word2idx['<UNK>'] = 3  # unknown token
    # with <START>, <END>, <UNK> tokens, without <PAD> token

    idx2word = {v: k for k, v in word2idx.items()}  # inverted vocabulary, for decoding, 11 -> 'word'

    if write_dict:
        print('All vocab:', len(all))
        print('Sub vocab: ', len(sub_all))
        print('Embedding vocab: ', len(embedding_words))
        print('Final vocab: ', len(embedded))

        f = open("data/models/data_dict.txt", "w", encoding='utf-8')
        for k, v in word2idx.items():
            f.write(k + '\n')
        f.close()

    return fdist, word2idx, idx2word


def count_padding_unknown(article_inputs, summary_inputs):
    article_unk = []
    summary_unk = []
    article_pad = []
    summary_pad = []

    for article in article_inputs:
        article = list(article)
        article_unk.append(article.count(3)/len(article))
        article_pad.append(article.count(0)/len(article))

    for summary in summary_inputs:
        summary = list(summary)
        summary_unk.append(summary.count(3)/len(summary))
        summary_pad.append(summary.count(0)/len(summary))

    return article_unk, summary_unk, article_pad, summary_pad


def pre_process(texts, word2idx, reverse):
    vectorized_texts = []

    for text in texts:  # vectorizes texts, array of tokens (words) -> array of ints (word2idx)
        text_vector = [word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in text]
        text_vector.insert(0, word2idx['<START>'])  # add <START> and <END> tokens to each summary/article
        text_vector.append(word2idx['<END>'])
        if reverse:
            vectorized_texts.append(list(reversed(text_vector)))
        else:
            vectorized_texts.append(text_vector)

    return vectorized_texts


def process_targets(summaries, word2idx):
    tmp_inputs = pre_process(summaries, word2idx, False)  # same as summaries_vectors, but with delay
    target_inputs = []  # ahead by one timestep, without start token
    for tmp in tmp_inputs:
        tmp.append(0)  # added 0 for padding, so the dimensions match
        target_inputs.append(tmp[1:])

    return target_inputs


def seq2seq_architecture(latent_size, vocabulary_size, embedding_matrix, batch_size, epochs, train_article,
                         train_summary, train_target):
    # encoder
    encoder_inputs = Input(shape=(None,), name='Encoder-Input')
    encoder_embeddings = Embedding(vocabulary_size + 1, 300, weights=[embedding_matrix],
                                   trainable=False, mask_zero=True, name='Encoder-Word-Embedding')(encoder_inputs)
    encoder_embeddings = BatchNormalization(name='Encoder-Batch-Normalization')(encoder_embeddings)
    _, state_h, state_c = LSTM(latent_size, return_state=True, dropout=0.2, recurrent_dropout=0.2,
                               name='Encoder-LSTM')(encoder_embeddings)
    encoder_states = [state_h, state_c]
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states, name='Encoder-Model')
    encoder_outputs = encoder_model(encoder_inputs)

    # decoder
    decoder_inputs = Input(shape=(None,), name='Decoder-Input')
    decoder_embeddings = Embedding(vocabulary_size + 1, 300, weights=[embedding_matrix],
                                   trainable=False, mask_zero=True, name='Decoder-Word-Embedding')(decoder_inputs)
    decoder_embeddings = BatchNormalization(name='Decoder-Batch-Normalization-1')(decoder_embeddings)
    decoder_lstm = LSTM(latent_size, return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                        name='Decoder-LSTM')
    decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_outputs)
    decoder_batchnorm = BatchNormalization(name='Decoder-Batch-Normalization-2')(decoder_lstm_outputs)
    decoder_outputs = Dense(vocabulary_size + 1, activation='softmax', name='Final-Output-Dense')(decoder_batchnorm)

    seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    seq2seq_model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy',
                          metrics=['sparse_categorical_accuracy'])
    seq2seq_model.summary()

    classes = [item for sublist in train_summary.tolist() for item in sublist]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(classes), classes)

    e_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min', restore_best_weights=True)
    history = seq2seq_model.fit(x=[train_article, train_summary], y=np.expand_dims(train_target, -1),
                                batch_size=batch_size, epochs=epochs, validation_split=0.1,
                                callbacks=[e_stopping], class_weight=class_weights)

    f = open("data/models/results.txt", "w", encoding="utf-8")
    f.write("LSTM \n layers: 1 \n latent size: " + str(latent_size) + "\n vocab size: " + str(vocabulary_size) + "\n")
    f.close()

    history_dict = history.history
    plot_loss(history_dict)

    # inference
    encoder_model = seq2seq_model.get_layer('Encoder-Model')

    decoder_inputs = seq2seq_model.get_layer('Decoder-Input').input
    decoder_embeddings = seq2seq_model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
    decoder_embeddings = seq2seq_model.get_layer('Decoder-Batch-Normalization-1')(decoder_embeddings)
    inference_state_h_input = Input(shape=(latent_size,), name='Hidden-State-Input')
    inference_state_c_input = Input(shape=(latent_size,), name='Cell-State-Input')

    lstm_out, lstm_state_h_out, lstm_state_c_out = seq2seq_model.get_layer('Decoder-LSTM')(
        [decoder_embeddings, inference_state_h_input, inference_state_c_input])
    decoder_outputs = seq2seq_model.get_layer('Decoder-Batch-Normalization-2')(lstm_out)
    dense_out = seq2seq_model.get_layer('Final-Output-Dense')(decoder_outputs)
    decoder_model = Model([decoder_inputs, inference_state_h_input, inference_state_c_input],
                          [dense_out, lstm_state_h_out, lstm_state_c_out])

    return encoder_model, decoder_model


def greedy_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len):
    states_value_h, states_value_c = encoder_model.predict(input_sequence)
    target_sequence = np.array(word2idx['<START>']).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, state_h, state_c = decoder_model.predict([target_sequence, states_value_h, states_value_c])

        predicted_word_index = np.argmax(candidates)
        if predicted_word_index == 0:
            predicted_word = '<END>'
        else:
            predicted_word = idx2word[predicted_word_index]

        prediction.append(predicted_word)

        if (predicted_word == '<END>') or (len(prediction) > max_len):
            stop_condition = True

        states_value_h = state_h
        states_value_c = state_c
        target_sequence = np.array(predicted_word_index).reshape(1, 1)  # previous character

    final = [x[0] for x in itertools.groupby(prediction[:-1])]  # remove <UNK> repetition
    return ' '.join(final)


def beam_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len):
    state_h, state_c = encoder_model.predict(input_sequence)
    # data[i] = [[prediction], score, stop_condition, state_h, state_c] = beam
    # target_sequence = prediction[-1] = last word

    data = [
        [[word2idx['<START>']], 0.0, False, state_h, state_c],
        [[word2idx['<START>']], 0.0, False, state_h, state_c],
        [[word2idx['<START>']], 0.0, False, state_h, state_c]]

    iteration = 1   # first iteration outside of loop
    probs, h, c = decoder_model.predict([np.array(word2idx['<START>']).reshape(1, 1), state_h, state_c])
    targets = np.argpartition(probs[0][0], -3)[-3:]

    for i, target in enumerate(targets):
        seq = data[i][0]
        seq.append(target)
        data[i][0] = seq
        data[i][1] = np.log(probs[0][0][target])
        data[i][3] = h
        data[i][4] = c

    while iteration < max_len:  # predict until max sequence length reached
        iteration += 1
        candidates = []

        for i, beam in enumerate(data):
            stop_condition = beam[2]
            if not stop_condition:
                target_sequence = np.array(beam[0][-1]).reshape(1, 1)  # previous word
                state_h = beam[3]
                state_c = beam[4]

                probs, h, c = decoder_model.predict([target_sequence, state_h, state_c])
                targets = np.argpartition(probs[0][0], -3)[-3:]  # predicted word indices
                score = beam[1]  # current score

                for target in targets:
                    candidates.append((i, target, score + np.log(probs[0][0][target])))  # update score

                data[i][3] = h  # update states
                data[i][4] = c

        # keep only top candidates, width of beam
        width = sum([1 if beam[2] == False else 0 for beam in data])

        if width == 0:  # stop, top candidates found
            break
        else:
            candidates.sort(key=lambda x: x[2])  # minimize score, ascending
            sorted = candidates[:width]

            for i, token, score in sorted:
                if token == 0 or token == word2idx['<END>']:  # stop predicting
                    data[i][1] = score
                    data[i][2] = True
                else:
                    data[i][1] = score
                    seq = data[i][0]
                    seq.append(token)
                    data[i][0] = seq

    top = []
    for beam in data:
        sequence = beam[0]
        prediction = []
        for token in sequence:
            prediction.append(idx2word[token])
        top.append(prediction)

    predictions = []
    for t in top:
        prediction = [x[0] for x in itertools.groupby(t[1:])]
        predictions.append(' '.join(prediction))

    return predictions


def evaluate(encoder_model, decoder_model, max_len, word2idx, idx2word, titles_test, summaries_test, articles_test):
    greedy_predictions = []
    beam_predictions = []

    for index in range(len(titles_test)):
        input_sequence = articles_test[index:index + 1]
        prediction_greedy = greedy_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len)
        prediction_beam = beam_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len)
        greedy_predictions.append(prediction_greedy)
        beam_predictions.append(prediction_beam)

        print(' '.join(titles_test[index:index + 1]))
        print(' '.join(summaries_test[index:index + 1][0]))
        print(prediction_greedy)
        print(prediction_beam, "\n")

    references = [' '.join(summary) for summary in summaries_test]
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=50,
                            length_limit_type='words',
                            apply_avg=False,
                            apply_best=True,
                            alpha=0.5,  # default F1 score
                            weight_factor=1.2,
                            stemming=True)

    f = open("data/models/results.txt", "a", encoding="utf-8")
    f.write('\n' + 'Greedy evaluation: ' + '\n')
    greedy_scores = evaluator.get_scores(greedy_predictions, references)
    for metric, results in sorted(greedy_scores.items(), key=lambda x: x[0]):
        score = prepare_results(metric, results['p'], results['r'], results['f'])
        print('Greedy evaluation: ' + score)
        f.write('\n' + score)

    f.write('\n' + '\n' + 'Beam evaluation: ' + '\n')
    beam_scores = evaluator.get_scores(references, beam_predictions)  # reversed precision and recall
    for metric, results in sorted(beam_scores.items(), key=lambda x: x[0]):
        score = prepare_results(metric, results['p'], results['r'], results['f'])
        print('Beam evaluation: ' + score)
        f.write('\n' + score)
    f.close()


# MAIN

titles, articles, summaries = read_data()
dataset_size = len(titles)
train = int(round(dataset_size * 0.98))
test = int(round(dataset_size * 0.02))

articles = clean_data(articles)
summaries = clean_data(summaries)
article_min_len, article_max_len, article_avg_len = analyze_data(articles)
summary_min_len, summary_max_len, summary_avg_len = analyze_data(summaries)

embeddings_index, n, d, embedding_words = load_embeddings()
all_tokens = list(itertools.chain(*articles)) + list(itertools.chain(*summaries))
fdist, word2idx, idx2word = build_vocabulary(all_tokens, embedding_words)
vocabulary_size = len(word2idx.items())

embedding_matrix = np.zeros((vocabulary_size + 1, 300))
embedding_matrix[1] = np.array(np.random.uniform(-1.0, 1.0, 300))  # <START>
embedding_matrix[2] = np.array(np.random.uniform(-1.0, 1.0, 300))  # <END>
embedding_matrix[3] = np.array(np.random.uniform(-1.0, 1.0, 300))  # <UNK>
for word, i in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None and i > 3:
        embedding_matrix[i] = embedding_vector

article_inputs = pre_process(articles, word2idx, True)
summary_inputs = pre_process(summaries, word2idx, False)
target_inputs = process_targets(summaries, word2idx)

article_inputs = pad_sequences(article_inputs, maxlen=article_max_len, padding='post')
summary_inputs = pad_sequences(summary_inputs, maxlen=summary_max_len, padding='post')
target_inputs = pad_sequences(target_inputs, maxlen=summary_max_len, padding='post')

article_unk, summary_unk, article_pad, summary_pad = count_padding_unknown(article_inputs, summary_inputs)

print('Dataset size (all/train/test): ', dataset_size, '/', train, '/', test)
print('Article lengths (min/max/avg): ', article_min_len, '/', article_max_len, '/', article_avg_len)
print('Summary lengths (min/max/avg): ', summary_min_len, '/', summary_max_len, '/', summary_avg_len)
print('Vocabulary size, without special tokens: ', vocabulary_size - 3)
print('Unknown (article/summary): ', round(sum(article_unk) / len(titles), 4), '/',
      round(sum(summary_unk) / len(titles), 4))
print('Padding (article/summary): ', round(sum(article_pad) / len(titles), 4), '/',
      round(sum(summary_pad) / len(titles), 4))

train_article = article_inputs[:train]
train_summary = summary_inputs[:train]
train_target = target_inputs[:train]
test_article = article_inputs[-test:]

latent_size = 512
batch_size = 16
epochs = 24

encoder_model, decoder_model = seq2seq_architecture(latent_size, vocabulary_size, embedding_matrix, batch_size, epochs,
                                                    train_article, train_summary, train_target)

evaluate(encoder_model, decoder_model, summary_max_len, word2idx, idx2word,
         titles[-test:], summaries[-test:], test_article)
