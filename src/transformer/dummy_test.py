import torch
from attention import MultiHeadSelfAttention

x = torch.randn(2, 4, 8)

attention = MultiHeadSelfAttention(
    d_model=8,
    num_heads=2
)

Q, K, V = attention(x)

print(Q.shape)
print(K.shape)
print(V.shape)