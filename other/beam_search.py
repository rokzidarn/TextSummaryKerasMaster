import numpy as np


def beam_search(encoder_model, decoder_model, input_sequence, word2idx, idx2word, max_len):
    state_h, state_c = encoder_model.predict(input_sequence)
    # data[i] = [[prediction], score, stop_condition, state_h, state_c] = beam
    # target_sequence = prediction[-1] = last word

    data = [
        [[word2idx['<START>']], 0.0, False, state_h, state_c],
        [[word2idx['<START>']], 0.0, False, state_h, state_c],
        [[word2idx['<START>']], 0.0, False, state_h, state_c]]
    iteration = 0

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
                    candidates.append((i, target[0], score + np.log(probs[target])))  # update score

                data[i][3] = h  # update states
                data[i][4] = c

        # keep only top candidates, width of beam
        width = sum([1 if beam[2] == False else 0 for beam in data])

        if width == 0:  # stop, top candidates found
            break
        else:
            sorted = candidates.sort(key=lambda x: x[2])[width:]  # minimize score, ascending
            for i, token, score in sorted:
                if token == word2idx['<PAD>'] or token == word2idx['<END>']:  # stop predicting
                    data[i][1] = score
                    data[i][2] = True
                else:
                    data[i][1] = score
                    data[i][0] = data[i][0].append(token)

    predictions = []
    for beam in data:
        sequence = beam[0]
        prediction = []
        for token in sequence:
            prediction.append(idx2word[token])
        predictions.append(prediction)

    return predictions
