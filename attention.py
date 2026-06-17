import torch
import torch.nn as nn
import torch.nn.functional as F

class selfAttention:
    def __init__(self, d_k):
        self.d_k = d_k

        self.w_q = nn.Linear(d_k, d_k, bias=False)
        self.w_k = nn.Linear(d_k, d_k,bias=False)
        self.w_v = nn.Linear(d_k,d_k,bias=False)
    
    def forward(self, x, d_k):
        """
        x shape: [batch_size, seq_len, d_k]
        """
        batch_size, seq_len, d_k = x.shape

        Q = self.w_q @ x
        K = self.w_k @ x
        V = self.w_v @ x

        scores = torch.matmul(Q,K.transpose(-2,-1))

        scores = scores / (self.d_k ** 0.5)

        attention_weight = F.softmax(scores, dim=-1)
        output = attention_weight @ V

        return output, attention_weight
    