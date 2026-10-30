import torch
import torch.nn as nn
import math

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff) # 升維
        self.act = nn.GELU() # GPT-2 用 GELU，Transformer原論文用 ReLU
        self.W2 = nn.Linear(d_ff, d_model) # 降維

    def forward(self,x):
        x=self.W1(x)  # [3, 4] -> [3, 16]
        x=self.act(x) # value filter
        x=self.W2(x) # [3,16] -> [3,4]
        return x

class MultiHeadAttention(nn.Module):
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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        attn_out, _ = self.attn(self.ln1(x))
        x = x + attn_out
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        d_ff=4*d_model
        self.token_emb=nn.Embedding(vocab_size,d_model)
        self.pos_emb=nn.Embedding(max_seq_len,d_model)
        self.blocks=nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab_size,bias=False)

        
    def forward(self,idx):
        B,T = idx.shape
        pos = torch.arange(T,device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)   # (B, T, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)    # (B, T, vocab_size)
        return logits
        
model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=1024)
x = torch.tensor([[464, 3797, 25365]])
logits = model(x)
print(logits.shape)