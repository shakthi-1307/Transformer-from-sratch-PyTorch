import torch 
import torch.nn as nn 

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_head = d_model//num_heads
        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)
        
        