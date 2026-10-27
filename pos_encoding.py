import torch
import math


vocab={"貓":0,"吃":1,"魚":2}
d_model=8
max_len = 10

def get_pos_encoding(max_len,d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pe = get_pos_encoding(max_len, d_model)
print(pe[:3]) # watch token encoding


torch.manual_seed(42)
embedding = torch.nn.Embedding(len(vocab), d_model)

def encode(sentence_tokens):
    ids = torch.tensor([vocab[t] for t in sentence_tokens])
    emb = embedding(ids)
    pos_enc = pe[: len(sentence_tokens)]
    return emb + pos_enc

cat_eat_fish = encode(["貓", "吃", "魚"])
fish_eat_cat = encode(["魚", "吃", "貓"])

print("貓吃魚:\n",cat_eat_fish)
print("魚吃貓:\n",fish_eat_cat)


long_pe=get_pos_encoding(5000,d_model)
print("pos encoding min value:", long_pe.min().item())
print("pos encoding max value:", long_pe.max().item())
