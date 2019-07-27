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


def seq2seq_architecture(latent_size, vocabulary_size):
    enc_in_id = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-IDs")  # None
    enc_in_mask = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-Masks")
    enc_in_segment = tf.keras.layers.Input(shape=(None, ), name="Encoder-Input-Segments")
    bert_encoder_inputs = [enc_in_id, enc_in_mask, enc_in_segment]

    encoder_embeddings = BertLayer(name='Encoder-Bert-Layer')(bert_encoder_inputs)
    encoder_embeddings = tf.keras.layers.BatchNormalization(name='Encoder-Batch-Normalization')(encoder_embeddings)
    _, state_h = tf.keras.layers.GRU(latent_size, return_state=True, name='Encoder-GRU')(encoder_embeddings)
    encoder_model = tf.keras.models.Model(inputs=bert_encoder_inputs, outputs=state_h, name='Encoder-Model')
    encoder_outputs = encoder_model(bert_encoder_inputs)

    dec_in_id = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-IDs")
    dec_in_mask = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-Masks")
    dec_in_segment = tf.keras.layers.Input(shape=(None,), name="Decoder-Input-Segments")
    bert_decoder_inputs = [dec_in_id, dec_in_mask, dec_in_segment]

    decoder_embeddings = BertLayer(name='Decoder-Bert-Layer')(bert_decoder_inputs)
    decoder_embeddings = tf.keras.layers.BatchNormalization(name='Decoder-Batchnormalization-1')(decoder_embeddings)
    decoder_gru = tf.keras.layers.GRU(latent_size, return_state=True, return_sequences=True, name='Decoder-GRU')
    decoder_gru_outputs, _ = decoder_gru(decoder_embeddings, initial_state=encoder_outputs)
    decoder_outputs = tf.keras.layers.BatchNormalization(name='Decoder-Batchnormalization-2')(decoder_gru_outputs)

    dec_out_id = tf.keras.layers.Dense(vocabulary_size, activation='softmax', name='Output-Dense-IDs')(decoder_outputs)
    dec_out_mask = tf.keras.layers.Dense(vocabulary_size, activation='softmax', name='Output-Dense-Masks')(decoder_outputs)
    dec_out_segment = tf.keras.layers.Dense(vocabulary_size, activation='softmax', name='Output-Dense-Segments')(decoder_outputs)

    seq2seq_model = tf.keras.models.Model(inputs=[enc_in_id, enc_in_mask, enc_in_segment,
                                                  dec_in_id, dec_in_mask, dec_in_segment],
                                          outputs=[dec_out_id, dec_out_mask, dec_out_segment])
    seq2seq_model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy',
                          metrics=['acc'])

    return seq2seq_model


def inference(model, latent_dim):
    encoder_model = model.get_layer('Encoder-Model')

    # latent_dim = model.get_layer('Decoder-Bert-Layer').output_shape[-1]  # 768
    dec_in_id = model.get_layer("Decoder-Input-IDs").input
    dec_in_mask = model.get_layer("Decoder-Input-Masks").input
    dec_in_segment = model.get_layer("Decoder-Input-Segments").input
    bert_decoder_inputs = [dec_in_id, dec_in_mask, dec_in_segment]
    decoder_embeddings = model.get_layer('Decoder-Bert-Layer')(bert_decoder_inputs)

    decoder_embeddings = model.get_layer('Decoder-Batchnormalization-1')(decoder_embeddings)
    gru_inference_state_input = tf.keras.layers.Input(shape=(latent_dim,), name='hidden_state_input')
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([decoder_embeddings, gru_inference_state_input])
    decoder_outputs = model.get_layer('Decoder-Batchnormalization-2')(gru_out)

    ids_out = model.get_layer('Output-Dense-IDs')(decoder_outputs)
    masks_out = model.get_layer('Output-Dense-Masks')(decoder_outputs)
    segments_out = model.get_layer('Output-Dense-Segments')(decoder_outputs)
    decoder_model = tf.keras.models.Model([dec_in_id, dec_in_mask, dec_in_segment, gru_inference_state_input],
                                          [ids_out, masks_out, segments_out, gru_state_out])

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, inputs, max_len, tokenizer):
    input_ids, input_masks, segment_ids = inputs
    states_value = encoder_model.predict([input_ids, input_masks, segment_ids])
    it = 1

    target_input = numpy.array(tokenizer.convert_tokens_to_ids(["[CLS]"])).reshape(1, 1)
    target_mask = numpy.array(1).reshape(1, 1)
    target_segment = numpy.array(it).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, masks, segments, state = decoder_model.predict([target_input, target_mask, target_segment, states_value])

        predicted_word_index = numpy.argmax(candidates)
        predicted_word = tokenizer.convert_ids_to_tokens([predicted_word_index])
        prediction.append(predicted_word)

        if (predicted_word == "[SEP]") or (len(prediction) > max_len):
            stop_condition = True

        states_value = state
        target_input = numpy.array(predicted_word_index).reshape(1, 1)

        if predicted_word == "." or predicted_word == "!" or predicted_word == "?":
            it += 1
        target_segment = numpy.array(it).reshape(1, 1)

    return prediction[:-1]
