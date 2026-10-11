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
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        scores = torch.matmul(Q,K.transpose(-2,-1))
        scores = scores/(self.d_head**0.5)
        
        return scores
    
    def split_heads(self,x):
        batch_size,seq_length,d_model = x.shape
        
        x = x.reshape(
            batch_size,
            seq_length,
            self.num_heads,
            self.d_head
        )
        
        x = x.transpose(1,2)
        
        return x 
    
    