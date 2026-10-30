from transformers import GPT2LMHeadModel
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


hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_state_dict = hf_model.state_dict()

# for name, tensor in list(hf_state_dict.items())[:5]:
#     print(name, tensor.shape)


def load_gpt2_weights(my_model, hf_state_dict, n_layer=12):
    sd = hf_state_dict

    # embedding
    my_model.token_embedding.weight.data = sd["transformer.wte.weight"]
    my_model.position_embedding.weight.data = sd["transformer.wpe.weight"]

    for i in range(n_layer):
        block = my_model.blocks[i]
        prefix = f"transformer.h.{i}."

        # LayerNorm
        block.ln1.weight.data = sd[prefix + "ln_1.weight"]
        block.ln1.bias.data   = sd[prefix + "ln_1.bias"]
        block.ln2.weight.data = sd[prefix + "ln_2.weight"]
        block.ln2.bias.data   = sd[prefix + "ln_2.bias"]

        # Attention:c_attn / c_proj need transpose
        block.attn.qkv_proj.weight.data = sd[prefix + "attn.c_attn.weight"].T
        block.attn.qkv_proj.bias.data   = sd[prefix + "attn.c_attn.bias"]
        block.attn.out_proj.weight.data = sd[prefix + "attn.c_proj.weight"].T
        block.attn.out_proj.bias.data   = sd[prefix + "attn.c_proj.bias"]

        # FFN:c_fc / c_proj all need transpose
        block.ffn.fc1.weight.data = sd[prefix + "mlp.c_fc.weight"].T
        block.ffn.fc1.bias.data   = sd[prefix + "mlp.c_fc.bias"]
        block.ffn.fc2.weight.data = sd[prefix + "mlp.c_proj.weight"].T
        block.ffn.fc2.bias.data   = sd[prefix + "mlp.c_proj.bias"]

    # 最终 LayerNorm
    my_model.final_ln.weight.data = sd["transformer.ln_f.weight"]
    my_model.final_ln.bias.data   = sd["transformer.ln_f.bias"]

    # 输出头:GPT-2 的输出头和 token embedding 权重共享(weight tying)
    my_model.output_head.weight.data = sd["transformer.wte.weight"]

    return my_model


config = {
    "vocab_size": 50257,    # 詞彙表大小
    "d_model": 768,        # 每個 token 的向量維度
    "n_heads": 12,         # 注意力頭數，每個頭 768 / 12 = 64 維
    "n_layers": 12,        # TransformerBlock 層數
    "max_seq_len": 1024,   # 最大序列長度
}

my_model = GPT(**config)

my_model = GPT(config)
my_model = load_gpt2_weights(my_model, hf_state_dict)
my_model.eval()

input_ids = torch.tensor([[464, 3797, 15070]]) # 編碼後的 "The cat eats" 之類的範例
with torch.no_grad():
    logits = my_model(input_ids)


next_token_id = logits[0, -1].argmax().item()
print(next_token_id)  # 現在這個預測才第一次"有意義"