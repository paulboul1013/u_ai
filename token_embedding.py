import torch
import torch.nn as nn


# token list have 30 token
vocab_size=30

# vector dim:very token use 8 dim vector to present
embedding_dim=8

# init 30 rows x 8 cols random table
embedding_layer=nn.Embedding(vocab_size,embedding_dim)

# 貓吃魚token id
token_ids=torch.tensor([3,15,27])

# query table: input token id,return vector
vectors=embedding_layer(token_ids)

print(vectors.shape)
# torch.Size([3, 8])
# 3個token,每個token都是8 dim vector

print(vectors)
