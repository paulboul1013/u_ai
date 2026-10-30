import torch
import torch.nn as nn

d_model = 4
d_ff = 4 * d_model  # GPT-2

# FFN(x) = W2 · Activation(W1 · x + b1) + b2
# Activation: ReLu,GELU
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


ffn=FeedForward(d_model,d_ff)

# 貓吃魚，LayerNorm之後的輸出，3個token，每個4維
x = torch.tensor([
    [0.9, -0.3, 0.1, -0.7],   # 貓
    [-0.4, 1.0, -0.6, 0.2],   # 吃
    [0.6, -0.5, 0.8, -0.4],   # 魚
])

out = ffn(x)
print(out.shape)  # torch.Size([3, 4])