import torch.nn as nn
import torch
import torch.nn.functional as F

class FeefForwardNetwork(nn.Module):
    def __init__(self, d_k:int, d_ff:int):

        self.fused_gate_up = nn.Linear(d_k, 2*d_ff, bias=False)

        self.down_proj = nn.Linear(d_k, 2*d_ff, bias=False)

    
        