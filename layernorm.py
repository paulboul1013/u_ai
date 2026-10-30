import torch
import torch.nn as nn

class layernorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # [d_model]
        self.beta = nn.Parameter(torch.zeros(d_model))   # [d_model]
        self.eps = eps

    # x_norm = (x - mean) / std
    # LN(x) = γ * x_norm + β
    def forward(self,x):
        # x: [batch, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)               # [batch, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False) # [batch, seq_len, 1]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # [batch, seq_len, d_model]
        return self.gamma * x_norm + self.beta            # [batch, seq_len, d_model]


torch.manual_seed(42)
x1 = torch.randn(1, 3, 8) # [batch=1, seq_len=3, d_model=8] after residual_connection output

my_ln = layernorm(d_model=8)
out_my = my_ln(x1)

official_ln = nn.LayerNorm(8)
out_official = official_ln(x1)

print("手寫layernorm输出 shape:", out_my.shape)        # [1, 3, 8]
print("官方layernorm输出 shape:", out_official.shape)  # [1, 3, 8]