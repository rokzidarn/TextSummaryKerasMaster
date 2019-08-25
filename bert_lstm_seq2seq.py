import os
import codecs
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from bert.tokenization import FullTokenizer
import rouge
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Input, LSTM, Dense, BatchNormalization
from tensorflow.python.keras.models import Model
# from keras.models import Model
# from keras.layers import Input, GRU, Dense, BatchNormalization


class BertLayer(Layer):
    '''
    pooled_output: the first CLS token after adding projection layer () with shape [batch_size, 768].
    sequence_output: all tokens output with shape [batch_size, max_length, 768].
    mean_pooling: mean pooling of all tokens output [batch_size, max_length, 768].
    '''

    def __init__(self, n_fine_tune_layers=10, tf_hub='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                 output_representation='sequence_output', trainable=False, **kwargs):

        self.n_fine_tune_layers = n_fine_tune_layers
        self.is_trainble = trainable
        self.output_size = 768
        self.tf_hub = tf_hub
        self.output_representation = output_representation
        self.supports_masking = True

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bert = hub.Module(
            self.tf_hub,
            trainable=self.is_trainble,
            name="{}_module".format(self.name)
        )

        variables = list(self.bert.variable_map.values())
        if self.is_trainble:
            # 1 first remove unused layers
            trainable_vars = [var for var in variables if not "/cls/" in var.name]

            if self.output_representation == "sequence_output" or self.output_representation == "mean_pooling":
                # 1 first remove unused pooled layers
                trainable_vars = [var for var in trainable_vars if not "/pooler/" in var.name]

            # Select how many layers to fine tune
            trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            # Add non-trainable weights
            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)

        else:
            for var in variables:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)

        if self.output_representation == "pooled_output":
            pooled = result["pooled_output"]

        elif self.output_representation == "mean_pooling":
            result_tmp = result["sequence_output"]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result_tmp, input_mask)

        elif self.output_representation == "sequence_output":

            pooled = result["sequence_output"]

        return pooled

    def compute_mask(self, inputs, mask=None):

        if self.output_representation == 'sequence_output':
            inputs = [K.cast(x, dtype="bool") for x in inputs]
            mask = inputs[1]

            return mask
        else:
            return None

    def compute_output_shape(self, input_shape):
        if self.output_representation == "sequence_output":
            return (input_shape[0][0], input_shape[0][1], self.output_size)
        else:
            return (input_shape[0][0], self.output_size)


def read_data():
    summaries = []
    articles = []
    titles = []

    ddir = 'data/news/'

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


def plot_training(history_dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = list(range(1, len(loss)+1))

    fig = plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('data/models/bert_seq2seq_train.png')

    fig = plt.figure()
    plt.plot(epochs, val_loss, 'r')
    plt.title('Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('data/models/bert_seq2seq_validation.png')


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


def tokenize_samples(tokenizer, samples, titles):
    words = []

    for i in range(len(samples)):
        tokens = tokenizer.tokenize(samples[i])
        tokens = tokens[:510]
        tokens.append("[SEP]")
        tokens = ["[CLS]"] + tokens
        words.append(tokens)
        # print(titles[i], len(tokens))

    max_len = len(max(words, key=len))
    min_len = len(min(words, key=len))
    avg_len = int(round(sum([len(x) for x in words])/len(samples)))

    return words, max_len, min_len, avg_len


def convert_sample(tokenizer, words, max_len):
    input_ids = tokenizer.convert_tokens_to_ids(words)
    segment_ids = [0] * len(input_ids)
    input_masks = [1] * len(input_ids)

    while len(input_ids) < max_len:
        input_ids.append(0)
        input_masks.append(0)
        segment_ids.append(0)

    return np.array(input_ids), np.array(input_masks), np.array(segment_ids)


def vectorize_features(tokenizer, samples, max_len):
    t_input_ids, t_input_masks, t_segment_ids = [], [], []

    for sample in samples:
        t_input_id, t_input_mask, t_segment_id = convert_sample(tokenizer, sample, max_len)
        t_input_ids.append(t_input_id)
        t_input_masks.append(t_input_mask)
        t_segment_ids.append(t_segment_id)

    return np.array(t_input_ids), np.array(t_input_masks), np.array(t_segment_ids)


def create_targets(summary_input_ids, summary_input_masks, summary_segment_ids):  # ahead by one timestep
    target_input_ids, target_masks, target_segment_ids = [], [], []

    for summary_input_id in summary_input_ids:
        target_input_id = numpy.append(summary_input_id[1:], 0)
        target_input_ids.append(target_input_id)

    for summary_input_mask in summary_input_masks:
        target_mask = numpy.append(summary_input_mask[1:], 0)
        target_masks.append(target_mask)

    for summary_segment_id in summary_segment_ids:
        target_segment_id = numpy.append(summary_segment_id[1:], 0)
        target_segment_ids.append(target_segment_id)

    return np.array(target_input_ids), np.array(target_masks), np.array(target_segment_ids)


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)


def seq2seq_architecture(latent_size, vocabulary_size, batch_size, epochs, sess, train_data_article, train_data_summary, train_data_target):
    (article_input_ids, article_input_masks, article_segment_ids) = train_data_article
    (summary_input_ids, summary_input_masks, summary_segment_ids) = train_data_summary
    (target_input_ids, target_masks, target_segment_ids) = train_data_target

    # encoder
    enc_in_id = Input(shape=(None, ), name="Encoder-Input-Ids")
    enc_in_mask = Input(shape=(None, ), name="Encoder-Input-Masks")
    enc_in_segment = Input(shape=(None, ), name="Encoder-Input-Segment-Ids")
    bert_encoder_inputs = [enc_in_id, enc_in_mask, enc_in_segment]

    encoder_embeddings = BertLayer(name='Encoder-Bert-Layer')(bert_encoder_inputs)
    encoder_embeddings = BatchNormalization(name='Encoder-Batch-Normalization')(encoder_embeddings)
    encoder_lstm = LSTM(latent_size, return_state=True, name='Encoder-LSTM')
    encoder_out, e_state_h, e_state_c = encoder_lstm(encoder_embeddings)
    encoder_states = [e_state_h, e_state_c]

    # decoder
    dec_in_id = Input(shape=(None,), name="Decoder-Input-Ids")
    dec_in_mask = Input(shape=(None,), name="Decoder-Input-Masks")
    dec_in_segment = Input(shape=(None,), name="Decoder-Input-Segment-Ids")
    bert_decoder_inputs = [dec_in_id, dec_in_mask, dec_in_segment]

    decoder_embeddings_layer = BertLayer(name='Decoder-Bert-Layer')
    decoder_embeddings = decoder_embeddings_layer(bert_decoder_inputs)
    decoder_batchnorm_layer = BatchNormalization(name='Decoder-Batch-Normalization-1')
    decoder_batchnorm = decoder_batchnorm_layer(decoder_embeddings)

    decoder_lstm = LSTM(latent_size, return_state=True, return_sequences=True, name='Decoder-LSTM')
    decoder_out, _, _ = decoder_lstm(decoder_batchnorm, initial_state=encoder_states)
    dense_batchnorm_layer = BatchNormalization(name='Decoder-Batch-Normalization-2')
    decoder_out_batchnorm = dense_batchnorm_layer(decoder_out)
    decoder_dense_id = Dense(vocabulary_size, activation='softmax', name='Dense-Id')
    dec_outputs_id = decoder_dense_id(decoder_out_batchnorm)

    seq2seq_model = Model(inputs=[enc_in_id, enc_in_mask, enc_in_segment,
                                  dec_in_id, dec_in_mask, dec_in_segment],
                          outputs=dec_outputs_id)
    seq2seq_model.summary()
    seq2seq_model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy',
                          metrics=['sparse_categorical_accuracy'])

    initialize_vars(sess)
    e_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')  # TODO: restore_best_weights=True
    history = seq2seq_model.fit([article_input_ids, article_input_masks, article_segment_ids,
                                 summary_input_ids, summary_input_masks, summary_segment_ids],
                                numpy.expand_dims(target_input_ids, -1),
                                epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[e_stopping])

    f = open("data/models/bert_results.txt", "w", encoding="utf-8")
    f.write("BERT \n layers: 1 \n latent size: " + str(latent_size) + "\n embeddings size: 768 \n")
    f.close()

    history_dict = history.history
    plot_training(history_dict)

    # inference
    encoder_model = Model(bert_encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_size,), name='Hidden-State-H')
    decoder_state_input_c = Input(shape=(latent_size,), name='Hidden-State-C')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embeddings_inf = decoder_embeddings_layer(bert_decoder_inputs)
    decoder_batchnorm_inf = decoder_batchnorm_layer(decoder_embeddings_inf)

    decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(decoder_batchnorm_inf, initial_state=decoder_states_inputs)
    decoder_out_batchnorm_inf = dense_batchnorm_layer(decoder_outputs_inf)
    decoder_states_inf = [state_h_inf, state_c_inf]
    decoder_outputs_id_inf = decoder_dense_id(decoder_out_batchnorm_inf)

    decoder_model = Model(inputs=bert_decoder_inputs + decoder_states_inputs,
                          outputs=[decoder_outputs_id_inf] + decoder_states_inf)

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, inputs, max_length_summary, tokenizer):
    input_ids, input_masks, segment_ids = inputs
    states_value = encoder_model.predict([input_ids, input_masks, segment_ids])

    target_input = np.zeros((1, 1))
    target_input[0][0] = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    target_mask = np.zeros((1, 1))
    target_mask[0][0] = 1
    target_segment = np.zeros((1, 1))
    target_segment[0][0] = 0

    prediction = []
    stop_condition = False
    previous = ''

    while not stop_condition:
        candidates, h, c = decoder_model.predict([target_input, target_mask, target_segment] + states_value)

        predicted_word_index = numpy.argmax(candidates[0, -1, :])
        # predicted_word_index = numpy.argsort(candidates)[-1]  # TODO: second candidate to resolve repetition
        predicted_word = tokenizer.convert_ids_to_tokens([predicted_word_index])[0]
        prediction.append(predicted_word)

        if (predicted_word == "[SEP]") or (predicted_word == "[PAD]") or (len(prediction) > max_length_summary):
            stop_condition = True

        target_input = np.zeros((1, 1))
        target_input[0][0] = predicted_word_index

        states_value = [h, c]
        previous = predicted_word

    return prediction[:-1]


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def evaluate(encoder_model, decoder_model, titles, summaries, test_data_article, max_length_summary):
    (article_input_ids, article_input_masks, article_segment_ids) = test_data_article
    predictions = []

    # testing
    for i in range(len(titles)):
        inputs = article_input_ids[i:i+1], article_input_masks[i:i+1], article_segment_ids[i:i+1]
        prediction = predict_sequence(encoder_model, decoder_model, inputs, max_length_summary, tokenizer)
        predictions.append(prediction)
        # print(prediction)

        f = open("data/news/predictions/" + titles[i] + ".txt", "w", encoding="utf-8")
        f.write(' '.join(prediction))
        f.close()

    # evaluation using ROUGE
    evaluator = rouge.Rouge(metrics=['rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=150,
                            length_limit_type='words',
                            apply_avg=False,
                            apply_best=True,
                            alpha=0.5,  # default F1 score
                            weight_factor=1.2,
                            stemming=True)

    all_hypothesis = [' '.join(prediction) for prediction in predictions]
    all_references = [' '.join(summary) for summary in summaries]
    scores = evaluator.get_scores(all_hypothesis, all_references)

    f = open("data/models/bert_results.txt", "a", encoding="utf-8")
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        score = prepare_results(metric, results['p'], results['r'], results['f'])
        print(score)
        f.write('\n' + score)
    f.close()


# MAIN

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU
sess = tf.Session()

tokenizer = create_tokenizer_from_hub_module(sess)
vocabulary_size = len(tokenizer.vocab)  # special end token <T>
titles, summaries, articles = read_data()
dataset_size = len(titles)
train = int(round(dataset_size * 0.95))
test = int(round(dataset_size * 0.05))

print("Dataset size all/train/test: ", dataset_size, train, test)
print("Vocabulary size: ", vocabulary_size)

article_tokens, max_len_article, min_len_article, avg_len_article = tokenize_samples(tokenizer, articles, titles)
summary_tokens, max_len_summary, min_len_summary, avg_len_summary = tokenize_samples(tokenizer, summaries, titles)

print("Article tokens max/avg/min length: ", max_len_article, avg_len_article, min_len_article)
print("Summary tokens max/avg/min length: ", max_len_summary, avg_len_summary, min_len_summary)

article_input_ids, article_input_masks, article_segment_ids = vectorize_features(tokenizer, article_tokens, max_len_article)
summary_input_ids, summary_input_masks, summary_segment_ids = vectorize_features(tokenizer, summary_tokens, max_len_summary)
target_input_ids, target_masks, target_segment_ids = create_targets(
    summary_input_ids, summary_input_masks, summary_segment_ids)

train_data_article = (article_input_ids[:train], article_input_masks[:train], article_segment_ids[:train])
train_data_summary = (summary_input_ids[:train], summary_input_masks[:train], summary_segment_ids[:train])
train_data_target = (target_input_ids[:train], target_masks[:train], target_segment_ids[:train])

test_data_article = (article_input_ids[-test:], article_input_masks[-test:], article_segment_ids[-test:])

latent_size = 384
batch_size = 12
epochs = 16

# training
encoder_model, decoder_model = seq2seq_architecture(latent_size, vocabulary_size, batch_size, epochs, sess,
                                                    train_data_article, train_data_summary, train_data_target)

# testing
evaluate(encoder_model, decoder_model, titles[-test:], summary_tokens[-test:], test_data_article, max_len_summary)
