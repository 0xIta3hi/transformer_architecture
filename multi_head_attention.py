import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention:
    def __init__(self, d_k, d_head):
        self.d_k = d_k
        self.d_head = d_head

        self.w_q = nn.Linear(d_k,d_k,bias=False)
        self.w_k = nn.Linear(d_k,d_k,bias=False)
        self.w_v = nn.Linear(d_k,d_k,bias=False)
        