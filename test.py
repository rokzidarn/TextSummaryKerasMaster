"""
words = tokenizer.tokenize("Pozdravljeni, gospod Rok. Dober dan tudi Vam!")
idx = tokenizer.convert_tokens_to_ids(words)
repeat = tokenizer.convert_ids_to_tokens(idx)
first = tokenizer.convert_ids_to_tokens([idx[0]])
print(len(words), idx, repeat, first)

special = "[CLS]"
sidx = tokenizer.convert_tokens_to_ids([special])
convert = tokenizer.convert_ids_to_tokens(sidx)
print(sidx, convert)

vectorized = convert_sample(words, 30)
print(vectorized)
"""