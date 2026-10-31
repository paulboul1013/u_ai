from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff) # 升維
        self.act = nn.GELU(approximate="tanh") # GPT-2 用 tanh 近似版 GELU，Transformer原論文用 ReLU
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
        self.d_model=d_model
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


@torch.no_grad()
def load_gpt2_weights(my_model, hf_state_dict, n_layer=12):
    sd = hf_state_dict

    # embedding
    my_model.token_emb.weight.copy_(sd["transformer.wte.weight"])
    my_model.pos_emb.weight.copy_(sd["transformer.wpe.weight"])

    for i in range(n_layer):
        block = my_model.blocks[i]
        prefix = f"transformer.h.{i}."

        # LayerNorm
        block.ln1.weight.copy_(sd[prefix + "ln_1.weight"])
        block.ln1.bias.copy_(sd[prefix + "ln_1.bias"])
        block.ln2.weight.copy_(sd[prefix + "ln_2.weight"])
        block.ln2.bias.copy_(sd[prefix + "ln_2.bias"])

        # Attention: HF 用 Conv1D，權重是 (in, out)，要轉置成 nn.Linear 的 (out, in)
        # c_attn 把 Q/K/V 合在一起 (768, 2304)，沿輸出維度切成三份
        q_w, k_w, v_w = sd[prefix + "attn.c_attn.weight"].split(my_model.d_model, dim=1)
        q_b, k_b, v_b = sd[prefix + "attn.c_attn.bias"].split(my_model.d_model, dim=0)
        block.attn.W_q.weight.copy_(q_w.T)
        block.attn.W_q.bias.copy_(q_b)
        block.attn.W_k.weight.copy_(k_w.T)
        block.attn.W_k.bias.copy_(k_b)
        block.attn.W_v.weight.copy_(v_w.T)
        block.attn.W_v.bias.copy_(v_b)
        block.attn.W_o.weight.copy_(sd[prefix + "attn.c_proj.weight"].T)
        block.attn.W_o.bias.copy_(sd[prefix + "attn.c_proj.bias"])

        # FFN: c_fc / c_proj 也都要轉置
        block.ffn.W1.weight.copy_(sd[prefix + "mlp.c_fc.weight"].T)
        block.ffn.W1.bias.copy_(sd[prefix + "mlp.c_fc.bias"])
        block.ffn.W2.weight.copy_(sd[prefix + "mlp.c_proj.weight"].T)
        block.ffn.W2.bias.copy_(sd[prefix + "mlp.c_proj.bias"])

    # 最終 LayerNorm
    my_model.ln_f.weight.copy_(sd["transformer.ln_f.weight"])
    my_model.ln_f.bias.copy_(sd["transformer.ln_f.bias"])

    # 輸出頭：GPT-2 的輸出頭和 token embedding 權重共享 (weight tying)
    my_model.head.weight.copy_(sd["transformer.wte.weight"])

    return my_model

def top_k_filter(logits, k):
    values, indices = torch.topk(logits, k)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits[indices] = values
    return filtered_logits

def generate(model, input_ids, max_new_tokens, temperature=0.8, top_k=50, eos_token_id=50256):
    model.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[0, -1] / temperature
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

    return input_ids

config = {
    "vocab_size": 50257,    # 詞彙表大小
    "d_model": 768,        # 每個 token 的向量維度
    "n_heads": 12,         # 注意力頭數，每個頭 768 / 12 = 64 維
    "n_layers": 12,        # TransformerBlock 層數
    "max_seq_len": 1024,   # 最大序列長度
}

my_model = GPT(**config)
my_model = load_gpt2_weights(my_model, hf_state_dict, n_layer=config["n_layers"])
my_model.eval()
hf_model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_ids = tokenizer("The cat eats", return_tensors="pt").input_ids
with torch.no_grad():
    logits = my_model(input_ids)
    hf_logits = hf_model(input_ids).logits

# 和 HuggingFace 原版比對，確認權重載入正確
print("max diff vs HF:", (logits - hf_logits).abs().max().item())

next_token_id = logits[0, -1].argmax().item()
print(next_token_id, repr(tokenizer.decode(next_token_id)))
