import numpy as np
import itertools
import copy


def beam_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len, raw):
    encoder_out, h1, c1, h2, c2 = encoder_model.predict(input_sequence)
    # data[i] = [[prediction], score, stop_condition, [encoder_out, h1, c1, h2, c2]] = beam

    data = [
        [[word2idx['<START>']], 0.0, False, [encoder_out, h1, c1, h2, c2]],
        [[word2idx['<START>']], 0.0, False, [encoder_out, h1, c1, h2, c2]],
        [[word2idx['<START>']], 0.0, False, [encoder_out, h1, c1, h2, c2]]]

    iteration = 1   # first iteration outside of loop
    probs, dh1, dc1, dh2, dc2 = decoder_model.predict([np.array(word2idx['<START>']).reshape(1, 1)] +
                                                      [encoder_out, h1, c1, h2, c2])
    targets = np.argpartition(probs[0][0], -3)[-3:]

    for i, target in enumerate(targets):
        beam = data[i]
        beam[0].append(target)
        beam[1] = np.log(probs[0][0][target])
        beam[3] = [encoder_out, dh1, dc1, dh2, dc2]
        data[i] = beam

    while iteration < max_len:  # predict until max sequence length reached
        iteration += 1
        candidates = []

        for i, beam in enumerate(data):
            stop_condition = beam[2]
            if not stop_condition:
                target_sequence = np.array(beam[0][-1]).reshape(1, 1)  # previous word
                states = beam[3]

                probs, dh1, dc1, dh2, dc2 = decoder_model.predict([target_sequence] + states)
                targets = np.argpartition(probs[0][0], -3)[-3:]  # predicted word indices
                score = beam[1]  # current score

                for i, target in enumerate(targets):
                    candidate = copy.deepcopy(beam)
                    candidate[0].append(target)
                    candidate[1] = score + np.log(probs[0][0][target])
                    candidate[3] = [encoder_out, dh1, dc1, dh2, dc2]
                    if target == 0 or target == word2idx['<END>']:
                        candidate[2] = True

                    candidates.append(candidate)  # update score, states

        candidates.sort(key=lambda x: x[1], reverse=True)  # minimize score, ascending

        next = 0
        for i in range(len(data)):
            if data[i][2] == False:
                data[i] = candidates[next]
                next += 1

    top = []
    for beam in data:
        sequence = beam[0]
        prediction = []
        for token in sequence:
            prediction.append(idx2word[token])
        top.append(prediction)

    predictions = []
    for t in top:
        final = [x[0] for x in itertools.groupby(t[1:-1])]
        predictions.append(' '.join(final))

    return predictions
