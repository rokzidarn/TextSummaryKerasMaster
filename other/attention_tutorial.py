from attention_layer import AttentionLayer

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf


def text_cleaner(text, num):
    ns = text.lower()
    ns = re.sub(r'\([^)]*\)', '', ns)
    ns = re.sub('"', '', ns)
    ns = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in ns.split(" ")])
    ns = re.sub(r"'s\b", "", ns)
    ns = re.sub("[^a-zA-Z]", " ", ns)
    ns = re.sub('[m]{2,}', 'mm', ns)

    if num == 0:
        tokens = [w for w in ns.split() if not w in stop_words]
    else:
        tokens = ns.split()

    long_words = []
    for i in tokens:
        if len(i) > 1:
            long_words.append(i)

    return (" ".join(long_words)).strip()


def seq2seq(x_tr, y_tr):
    tf.keras.backend.clear_session()
    latent_dim = 300
    embedding_dim = 100

    # MODEL

    encoder_inputs = tf.keras.layers.Input(shape=(max_text_len,))
    enc_emb = tf.keras.layers.Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)
    encoder_lstm1 = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4,
                                         recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
    encoder_lstm2 = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4,
                                         recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
    encoder_lstm3 = tf.keras.layers.LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4,
                                         recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    dec_emb_layer = tf.keras.layers.Embedding(y_voc, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4,
                                        recurrent_dropout=0.2)
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    decoder_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.summary()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=10,
                        batch_size=64)

    # INFERENCE

    encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
    decoder_state_input_h = tf.keras.layers.Input(shape=(latent_dim,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(latent_dim,))
    decoder_hidden_state_input = tf.keras.layers.Input(shape=(max_text_len, latent_dim))

    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2,
                                                        initial_state=[decoder_state_input_h, decoder_state_input_c])

    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = tf.keras.layers.Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])

    return encoder_model, decoder_model


def seq2summary(input_seq, target_word_index, reverse_target_word_index):
    ns = ''
    for i in input_seq:
        if i != 0 and i != target_word_index['sostok'] and i != target_word_index['eostok']:
            ns = ns + reverse_target_word_index[i] + ' '

    return ns


def seq2text(input_seq, reverse_source_word_index):
    ns = ''
    for i in input_seq:
        if i != 0:
            ns = ns + reverse_source_word_index[i] + ' '

    return ns


def decode_sequence(input_seq, encoder_model, decoder_model, target_word_index, reverse_target_word_index,
                    max_summary_len):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= max_summary_len - 1:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c

    return decoded_sentence


# MAIN
# TODO: https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/master/How_to_build_own_text_summarizer_using_deep_learning.ipynb

data = pd.read_csv("../data/reviews.csv", nrows=10000)
data.drop_duplicates(subset=['Text'], inplace=True)  # dropping duplicates
data.dropna(axis=0, inplace=True)  # dropping na

print(data.info())

contraction_mapping = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
    "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
    "I've": "I have", "i'd": "i would",
    "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so as",
    "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
    "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
    "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
    "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"
}

stop_words = set(stopwords.words('english'))

cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t, 0))

cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(text_cleaner(t, 1))

data['cleaned_text'] = cleaned_text
data['cleaned_summary'] = cleaned_summary

data.replace('', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)

text_word_count = []
summary_word_count = []

for i in data['cleaned_text']:
    text_word_count.append(len(i.split()))

for i in data['cleaned_summary']:
    summary_word_count.append(len(i.split()))

length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})
length_df.hist(bins=30)

cnt = 0
for i in data['cleaned_summary']:
    if len(i.split()) <= 8:
        cnt = cnt + 1

max_text_len = 30
max_summary_len = 8

cleaned_text = np.array(data['cleaned_text'])
cleaned_summary = np.array(data['cleaned_summary'])

short_text = []
short_summary = []

for i in range(len(cleaned_text)):
    if len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len:
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])

df = pd.DataFrame({'text': short_text, 'summary': short_summary})
df['summary'] = df['summary'].apply(lambda x: 'sostok ' + x + ' eostok')

x_tr, x_val, y_tr, y_val = train_test_split(np.array(df['text']), np.array(df['summary']), test_size=0.1,
                                            random_state=0, shuffle=True)

x_tokenizer = tf.keras.preprocessing.text.Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

thresh = 4
cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in x_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if value < thresh:
        cnt = cnt + 1
        freq = freq + value

x_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=tot_cnt - cnt)
x_tokenizer.fit_on_texts(list(x_tr))

x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

x_tr = tf.keras.preprocessing.sequence.pad_sequences(x_tr_seq, maxlen=max_text_len, padding='post')
x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')
x_voc = x_tokenizer.num_words + 1

y_tokenizer = tf.keras.preprocessing.text.Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

thresh = 6
cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if value < thresh:
        cnt = cnt + 1
        freq = freq + value

y_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=tot_cnt - cnt)
y_tokenizer.fit_on_texts(list(y_tr))

y_tr_seq = y_tokenizer.texts_to_sequences(y_tr)
y_val_seq = y_tokenizer.texts_to_sequences(y_val)

y_tr = tf.keras.preprocessing.sequence.pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val = tf.keras.preprocessing.sequence.pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')
y_voc = y_tokenizer.num_words + 1

ind = []
for i in range(len(y_tr)):
    cnt = 0
    for j in y_tr[i]:
        if j != 0:
            cnt = cnt + 1
    if cnt == 2:
        ind.append(i)

y_tr = np.delete(y_tr, ind, axis=0)
x_tr = np.delete(x_tr, ind, axis=0)

ind = []
for i in range(len(y_val)):
    cnt = 0
    for j in y_val[i]:
        if j != 0:
            cnt = cnt + 1
    if cnt == 2:
        ind.append(i)

y_val = np.delete(y_val, ind, axis=0)
x_val = np.delete(x_val, ind, axis=0)

encoder_model, decoder_model = seq2seq(x_tr, y_tr)

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

for i in range(0, 100):
    print("Review:", seq2text(x_tr[i], reverse_source_word_index))
    print("Original summary:", seq2summary(y_tr[i], target_word_index, reverse_target_word_index))
    print("Predicted summary:", decode_sequence(x_tr[i].reshape(1, max_text_len), encoder_model, decoder_model,
                                                target_word_index, reverse_target_word_index, max_summary_len))
    print("\n")
