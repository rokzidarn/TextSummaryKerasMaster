import os
import codecs
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from bert.tokenization import FullTokenizer


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=0, **kwargs):
        # TODO: 3 = low loss+high acc (less trainable params), 0 = high loss+low acc (more trainable params)
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1",
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables
        # trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name and not "/pooler/" in var.name]
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [tf.keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "sequence_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size


def plot_training(history_dict, epochs):
    loss = history_dict['loss']

    fig = plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    fig.savefig('data/models/bert.png')


def read_data():
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


def create_tokenizer_from_hub_module(sess):
    bert_module = hub.Module("https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1")
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def tokenize_samples(tokenizer, samples):  # TODO
    words = []

    for sample in samples:
        words.append(tokenizer.tokenize(sample))

    max_seq_length = len(max(words, key=len))

    return words, max_seq_length + 6  # TODO: count number of [SEP] tokens needed


def convert_sample(tokenizer, words, max_seq_length):
    it = 1
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")  # 101: sentence start token
    segment_ids.append(it)

    for token in words:
        tokens.append(token)
        segment_ids.append(it)
        if token == "." or token == "!" or token == "?":  # check segments (sentence splitting), first sentence == 1
            tokens.append("[SEP]")  # 102: sentence end token
            segment_ids.append(it)
            it += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # the mask has 1 for real tokens and 0 for padding tokens, only real tokens are attended to
    input_masks = [1] * len(input_ids)

    # zero-pad up to the sequence length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_masks.append(0)
        segment_ids.append(0)

    return np.array(input_ids), np.array(input_masks), np.array(segment_ids)


def vectorize_features(tokenizer, samples, max_seq_length):
    input_ids, input_masks, segment_ids = [], [], []

    for sample in samples:
        input_id, input_mask, segment_id = convert_sample(tokenizer, sample, max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)

    return np.array(input_ids), np.array(input_masks), np.array(segment_ids)


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


def seq2seq_architecture(latent_size, vocabulary_size):
    enc_in_id = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-ids")  # None
    enc_in_mask = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-Masks")
    enc_in_segment = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-Segment-ids")
    bert_encoder_inputs = [enc_in_id, enc_in_mask, enc_in_segment]

    encoder_embeddings = BertLayer(name='Encoder-Bert-Layer')(bert_encoder_inputs)
    encoder_embeddings = tf.keras.layers.BatchNormalization(name='Encoder-Batch-Normalization')(encoder_embeddings)
    _, state_h = tf.keras.layers.GRU(latent_size, return_state=True, name='Encoder-GRU')(encoder_embeddings)
    encoder_model = tf.keras.models.Model(inputs=bert_encoder_inputs, outputs=state_h, name='Encoder-Model')
    encoder_outputs = encoder_model(bert_encoder_inputs)

    dec_in_id = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-ids")
    dec_in_mask = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-Masks")
    dec_in_segment = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-Segment-ids")
    bert_decoder_inputs = [dec_in_id, dec_in_mask, dec_in_segment]

    decoder_embeddings = BertLayer(name='Decoder-Bert-Layer')(bert_decoder_inputs)
    decoder_embeddings = tf.keras.layers.BatchNormalization(name='Decoder-Batchnormalization-1')(decoder_embeddings)
    decoder_gru = tf.keras.layers.GRU(latent_size, return_state=True, return_sequences=True, name='Decoder-GRU')
    decoder_gru_outputs, _ = decoder_gru(decoder_embeddings, initial_state=encoder_outputs)
    decoder_outputs = tf.keras.layers.BatchNormalization(name='Decoder-Batchnormalization-2')(decoder_gru_outputs)
    decoder_outputs = tf.keras.layers.Dense(vocabulary_size, activation='softmax', name='Final-Output-Dense')(decoder_outputs)

    seq2seq_model = tf.keras.models.Model(inputs=[enc_in_id, enc_in_mask, enc_in_segment,
                                                  dec_in_id, dec_in_mask, dec_in_segment], outputs=decoder_outputs)
    seq2seq_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy',
                          metrics=['sparse_categorical_accuracy'])

    return seq2seq_model


def inference(model, latent_dim):
    encoder_model = model.get_layer('Encoder-Model')

    # latent_dim = model.get_layer('Decoder-Bert-Layer').output_shape[-1]  # 768
    dec_in_id = model.get_layer("Decoder-Input-ids").input
    dec_in_mask = model.get_layer("Decoder-Input-Masks").input
    dec_in_segment = model.get_layer("Decoder-Input-Segment-ids").input
    bert_decoder_inputs = [dec_in_id, dec_in_mask, dec_in_segment]
    decoder_embeddings = model.get_layer('Decoder-Bert-Layer')(bert_decoder_inputs)

    decoder_embeddings = model.get_layer('Decoder-Batchnormalization-1')(decoder_embeddings)
    gru_inference_state_input = tf.keras.layers.Input(shape=(latent_dim,), name='Hidden-State-Input')
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([decoder_embeddings, gru_inference_state_input])
    decoder_outputs = model.get_layer('Decoder-Batchnormalization-2')(gru_out)
    dense_out = model.get_layer('Final-Output-Dense')(decoder_outputs)
    decoder_model = tf.keras.models.Model([dec_in_id, dec_in_mask, dec_in_segment, gru_inference_state_input],
                                          [dense_out, gru_state_out])

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, inputs, max_len, tokenizer):
    input_ids, input_masks, segment_ids = inputs
    states_value = encoder_model.predict([input_ids, input_masks, segment_ids])
    it = 1

    # target_input = numpy.array(tokenizer.convert_tokens_to_ids(["[CLS]"])).reshape(1, 1)
    target_input = numpy.array(tokenizer.convert_tokens_to_ids(["[CLS]"])[0]).reshape(1, 1)
    target_mask = numpy.array(1).reshape(1, 1)
    target_segment = numpy.array(it).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, state = decoder_model.predict([target_input, target_mask, target_segment, states_value])

        predicted_word_index = numpy.argmax(candidates)
        # predicted_word_index = numpy.argsort(candidates)[-1]  # same as argmax
        predicted_word = tokenizer.convert_ids_to_tokens([predicted_word_index])[0]
        prediction.append(predicted_word)

        if (predicted_word == "[SEP]") or (len(prediction) > max_len):
            stop_condition = True

        states_value = state
        target_input = numpy.array(predicted_word_index).reshape(1, 1)
        target_mask = numpy.array(1).reshape(1, 1)

        if predicted_word == "." or predicted_word == "!" or predicted_word == "?":
            it += 1
        target_segment = numpy.array(it).reshape(1, 1)

    return prediction[:-1]


# MAIN

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU
sess = tf.Session()
tokenizer = create_tokenizer_from_hub_module(sess)
vocabulary_size = len(tokenizer.vocab)

titles, summaries, articles = read_data()
article_tokens, max_len_article = tokenize_samples(tokenizer, articles)
summary_tokens, max_len_summary = tokenize_samples(tokenizer, summaries)

article_input_ids, article_input_masks, article_segment_ids = vectorize_features(tokenizer, article_tokens, max_len_article)
summary_input_ids, summary_input_masks, summary_segment_ids = vectorize_features(tokenizer, summary_tokens, max_len_summary)
target_input_ids, target_masks, target_segment_ids = create_targets(summary_input_ids, summary_input_masks, summary_segment_ids)

latent_size = 768
batch_size = 1
epochs = 4

seq2seq_model = seq2seq_architecture(latent_size, vocabulary_size)
seq2seq_model.summary()

initialize_vars(sess)

seq2seq_model.fit([article_input_ids, article_input_masks, article_segment_ids,
                   summary_input_ids, summary_input_masks, summary_segment_ids],
                  numpy.expand_dims(target_input_ids, -1), epochs=epochs, batch_size=batch_size)

encoder_model, decoder_model = inference(seq2seq_model, latent_size)

for i in range(3):
    inputs = article_input_ids[i:i+1], article_input_masks[i:i+1], article_segment_ids[i:i+1]
    prediction = predict_sequence(encoder_model, decoder_model, inputs, max_len_summary, tokenizer)

    print('-')
    print('Summary:', summary_tokens[i])
    print('Prediction:', prediction)
