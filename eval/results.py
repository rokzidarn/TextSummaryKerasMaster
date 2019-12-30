import sys
import os
import rouge
import warnings
import statistics
warnings.simplefilter(action='ignore', category=FutureWarning)
from bert_score import score


def read_article(file):
    article = open('../data/sta/articles/' + file + '.txt', 'r', encoding='utf-8').read()
    return article


def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def best_rouge_score(file, article, summary, greedy, beam):
    candidates = [beam[0][1:-1], beam[1][2:-1], beam[2][2:-2]]

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=False,
                            apply_best=True,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=False)

    rouge_scores = []

    for candidate in candidates:
        all_hypothesis = [candidate]
        all_references = [summary]
        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            # print(prepare_results(metric, results['p'], results['r'], results['f']))
            rouge_scores.append([candidate, round(results['f'] * 100.0, 2)])
            break

    rouge_scores.sort(key=lambda x: x[1], reverse=True)

    return rouge_scores[0][0], rouge_scores[0][1]


# MAIN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

log = open('log.txt', 'a', encoding='utf-8')
# sys.stdout = log

results = 'best.txt'
data = []
refs = []
hyps = []
f1_rouge = []

lines = open(results, 'r', encoding='utf-8').read().splitlines()
length = len(lines)

for i in range(0, length, 5):
    file = lines[i]
    article = read_article(file)
    summary = lines[i+1]
    greedy = lines[i+2]
    beam = lines[i+3][1:-1].split(',')
    final, rouge_score = best_rouge_score(file, article, summary, greedy, beam)
    data.append([file, article, summary, greedy, final, rouge_score, 0.0])
    f1_rouge.append(rouge_score)
    refs.append(summary)
    hyps.append(final)

p, r, f1_bert = score(refs, hyps, model_type='bert-base-multilingual-cased')

for i in range(len(data)):
    data[i][6] = round(f1_bert[i].item(), 3)

data.sort(key=lambda x: x[5])  # reverse=True

for d in data:
    print(d[0], '\n', d[1], d[2], '\n', d[4], '\n', d[5], '\n', d[6], '\n')
    print('------------------------------------------------------------\n')

print(f"ROUGESCORE: {statistics.mean(f1_rouge):.3f}", '\n')
print(f"BERTSCORE: {f1_bert.mean():.3f}", '\n')
