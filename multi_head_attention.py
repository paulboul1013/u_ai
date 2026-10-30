import torch
import torch.nn as nn
import math


class multi_head_attention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads==0 #"must divied"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self,x):
        batch_size,seq_len,_=x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # split to multi-head (batch,seq_len,d_model) -> (batch,num_heads,seq_len,d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # scale dot attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # concat multi head (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.W_o(attn_output)
        return output, attn_weights


vocab={"貓": 0, "吃": 1, "魚": 2}
d_model = 8
num_heads = 2

torch.manual_seed(42)
embedding = nn.Embedding(len(vocab), d_model)

sentence = ["貓","吃","魚"]
token_ids = torch.tensor([[vocab[w] for w in sentence]])
x = embedding(token_ids)  # (1, 3, 8)

mha = multi_head_attention(d_model=d_model, num_heads=num_heads)
output, attn_weights = mha(x)

print("輸入 shape:", x.shape)              # torch.Size([1, 3, 8])
print("輸出 shape:", output.shape)          # torch.Size([1, 3, 8])
print("注意力權重 shape:", attn_weights.shape)  # torch.Size([1, 2, 3, 3])