import torch
import torch.nn.functional as F

# 貓吃魚,d_model=8
# X: after pass into Embedding + Pos encoding value
X=torch.randn(3,8)

d_model=8
d_k=8 # single attention head

W_q = torch.nn.Linear(d_model, d_k, bias=False)
W_k = torch.nn.Linear(d_model, d_k, bias=False)
W_v = torch.nn.Linear(d_model, d_k, bias=False)

Q = W_q(X)   # (3, 8)
K = W_k(X)   # (3, 8)
V = W_v(X)   # (3, 8)


# cal similarity 
scores=  Q @ K.T  #(3,3)

# scale
scores = scores / (d_k ** 0.5)

# map into softmax
attn_weights = F.softmax(scores, dim=-1)  #(3,3)

# multi value weight 
output = attn_weights @ V # (3, 8)

print("attention weight matrix:\n",attn_weights)
print("output:\n",output)


