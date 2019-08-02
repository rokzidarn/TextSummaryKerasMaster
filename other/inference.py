import numpy
from keras.models import Model
from keras.layers import Input


def inference(model):
    encoder_model = model.get_layer('Encoder-Model')

    latent_dim = model.get_layer('Decoder-Word-Embedding').output_shape[-1]  # gets embedding size, not latent size
    decoder_inputs = model.get_layer('Decoder-Input').input
    decoder_embeddings = model.get_layer('Decoder-Word-Embedding')(decoder_inputs)
    decoder_embeddings = model.get_layer('Decoder-Batchnormalization-1')(decoder_embeddings)
    gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([decoder_embeddings, gru_inference_state_input])
    decoder_outputs = model.get_layer('Decoder-Batchnormalization-2')(gru_out)
    dense_out = model.get_layer('Final-Output-Dense')(decoder_outputs)
    decoder_model = Model([decoder_inputs, gru_inference_state_input], [dense_out, gru_state_out])

    return encoder_model, decoder_model


def predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len):
    # encode the input as state vectors
    states_value = encoder_model.predict(input_sequence)
    # populate the first character of target sequence with the start character
    target_sequence = numpy.array(word2idx['<START>']).reshape(1, 1)

    prediction = []
    stop_condition = False

    while not stop_condition:
        candidates, state = decoder_model.predict([target_sequence, states_value])

        predicted_word_index = numpy.argmax(candidates)  # sample token (word), greedy search
        predicted_word = idx2word[predicted_word_index]
        prediction.append(predicted_word)

        # exit condition, either hit max length or find stop character
        if (predicted_word == '<END>') or (len(prediction) > max_len):
            stop_condition = True

        states_value = state  # update states
        target_sequence = numpy.array(predicted_word_index).reshape(1, 1)  # update target sequence, previous character

    return prediction[:-1]


"""
In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
"""