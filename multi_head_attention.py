import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention:
    def __init__(self, d_k, d_head):
        assert self.d_k % d_head == 0
        self.d_k = d_k
        self.d_head = d_head
        self.d_model = d_k // d_head
        self.w_q = nn.Linear(d_k,d_k,bias=False)
        self.w_k = nn.Linear(d_k,d_k,bias=False)
        self.w_v = nn.Linear(d_k,d_k,bias=False)
        self.w_o = nn.Linear(d_k,d_k,bias=False) # final projection layer
    
    def forward(self, x, mask=None):
        B,T, d_k = x.shape

        Q = self.w_q @ x
        K = self.w_k @ x
        V = self.w_v @ x

        # now we split these
        q = q.view(B, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        score = torch.matmul(q,k.transpose(-2,-1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(score, dim=-1)

        output = attention_weights @ v

        context = context.purmute(0,2,1,3).contiguous()

        concat_output = context.view(B, T, self.d_model)
        
        # 5. Final linear projection mix
        return self.W_o(concat_output), attention_weights
    
if __name__ == "__main__":
    # Standard Transformer config
    mha = MultiHeadAttention(d_k=512, d_head=8)
    sample_tokens = torch.randn(2, 5, 512) # B=2, T=5, d_model=512
    
    out, weights = mha(sample_tokens)
    print("--- MULTI-HEAD ATTENTION COMPILED SUCCESSFULLY ---")
    print("Input Tensor Shape:      ", sample_tokens.shape)
    print("Final Output Shape:     ", out.shape)
    print("Attention Weights Shape: ", weights.shape)  # Should be [2, 8, 5, 5]
        
