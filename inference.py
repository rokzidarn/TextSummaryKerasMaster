import numpy
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
from pickle import load
import rouge
import codecs

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

        predicted_word_index = numpy.argmax(candidates)  # greedy search
        predicted_word = idx2word[predicted_word_index]
        prediction.append(predicted_word)

        # exit condition, either hit max length or find stop character
        if (predicted_word == '<END>') or (len(prediction) > max_len):
            stop_condition = True

        states_value = state
        target_sequence = numpy.array(predicted_word_index).reshape(1, 1)  # previous character

    return prediction[:-1]


def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


# MAIN

# inference
model = load_model('data/models/gru_seq2seq_model.h5')  # loads saved model
[titles, X_article, summaries_clean, word2idx, idx2word, max_length_summary] = \
    load(open('data/models/serialized_data.pkl', 'rb'))  # loads serialized data
encoder_model, decoder_model = inference(model)
ddir = 'data/test/'

# testing
predictions = []
print("RESULTS: ")
for index in range(len(titles)):
    input_sequence = X_article[index:index+1]
    prediction = predict_sequence(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_length_summary)
    predictions.append(prediction)

    print('')
    print('Title:', titles[index])
    print('Summary:', summaries_clean[index])
    print('Prediction:', prediction)

    with codecs.open(ddir+'predictions/'+titles[index]+'.txt', 'w', encoding='utf8') as f:
        f.write("{}\n".format(prediction))

# evaluation using ROUGE
aggregator = 'Best'
evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                        max_n=4,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=False,
                        apply_best=True,
                        alpha=0.5,  # default F1 score
                        weight_factor=1.2,
                        stemming=True)

all_hypothesis = [' '.join(prediction) for prediction in predictions]
all_references = [' '.join(summary) for summary in summaries_clean]

scores = evaluator.get_scores(all_hypothesis, all_references)
# https://pypi.org/project/py-rouge/

print()
print('ROUGE evaluation: ')
for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print('\n', prepare_results(results['p'], results['r'], results['f']))
