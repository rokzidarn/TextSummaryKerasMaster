import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

text = "Kos"
marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)

print(tokenized_text)
print(indexed_tokens)
print(len(tokenized_text))

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)

layer_i = 0
batch_i = 0
token_i = 0

token_embeddings = []

for token_i in range(len(tokenized_text)):
    hidden_layers = []

    for layer_i in range(len(encoded_layers)):
        vec = encoded_layers[layer_i][batch_i][token_i]
        hidden_layers.append(vec)

    token_embeddings.append(hidden_layers)

token_vecs_sum = []

for token in token_embeddings:
    sum_vec = torch.sum(torch.stack(token)[-4:], 0)
    token_vecs_sum.append(sum_vec)

print((len(token_vecs_sum), len(token_vecs_sum[0])))
print(token_vecs_sum[1][:15])
