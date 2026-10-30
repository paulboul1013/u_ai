import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, d_model]，例如 [3, 8] 對應"貓吃魚"
        return x + self.sublayer(x) #  [3, 8]，子層輸出直接加在原始x上

# 模擬"貓吃魚"：seq_len=3, d_model=8
X = torch.randn(3, 8) # after embedding+positon 

# fake multi head output
attn_output = torch.randn(3, 8)

# residual connection
output = X + attn_output # [3, 8]

print(X.shape, attn_output.shape, output.shape)
# torch.Size([3, 8]) torch.Size([3, 8]) torch.Size([3, 8])
