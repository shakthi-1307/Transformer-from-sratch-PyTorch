import torch
from attention import MultiHeadSelfAttention

x = torch.randn(2, 4, 8)

attention = MultiHeadSelfAttention(
    d_model=8,
    num_heads=2
)

context = attention(x)
print("Context shape:", context.shape)