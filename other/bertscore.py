from bert_score import score, plot_example
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

with open("../data/models/hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("../data/models/refs.txt") as f:
    refs = [line.strip() for line in f]

P, R, F1 = score(cands, refs, lang='en')

print(f"System level F1 score: {F1.mean():.3f}")

plt.hist(F1, bins=20)
plt.show()

print(cands[0])
print(f'P={P[0]:.6f} R={R[0]:.6f} F={F1[0]:.6f}')
plot_example(cands[0], refs[0], lang="en")
