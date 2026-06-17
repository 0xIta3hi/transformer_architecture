import torch
import torch.nn as nn
import torch.nn.functional as F

class selfAttention:
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.w_q = nn.Linear(d_model,d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model,d_model,bias=False)

    def forward(self, x, mask=None):
        """
        x shape: [batch_size, seq_len, d_model].
        """
        batch_size, seq_len, d_model = x.shape

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        