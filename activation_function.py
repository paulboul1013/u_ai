import torch
import torch.nn as nn


# 貓吃魚，3個token，d_model=4，升維到4*d_model=16
x = torch.randn(3, 4)  # shape: [seq_len=3, d_model=4]


W1 = nn.Linear(4, 16) # 升維
W2 = nn.Linear(16, 4) # 降維

relu = nn.ReLU()
gelu = nn.GELU()

hidden = W1(x)          # shape: [3, 16]

hidden_relu = relu(hidden)
hidden_gelu = gelu(hidden)

out_relu = W2(hidden_relu)   # shape: [3, 4]
out_gelu = W2(hidden_gelu)   # shape: [3, 4]

print("ReLU:", (hidden_relu == 0).sum().item(), "/", hidden_relu.numel())
print("GELU:", ((hidden_gelu.abs() < 0.01) & (hidden_gelu != 0)).sum().item())
