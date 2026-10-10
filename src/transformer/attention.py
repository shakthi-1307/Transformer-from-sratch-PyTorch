import torch 
import torch.nn as nn 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_head = d_model//num_heads
        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)
        
    def forward(self,x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        return Q,K,V
    