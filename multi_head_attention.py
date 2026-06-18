import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_head):
        """
        d_k: Total embedding dimension (e.g., 512)
        d_head: Number of attention heads (e.g., 8)
        """
        super().__init__()
        # FIX: Assert directly on the incoming local arguments before binding them to self
        assert d_k % d_head == 0, "Total dimension (d_k) must be perfectly divisible by number of heads (d_head)!"
        
        self.d_k = d_k              # Total dimension highway width
        self.d_head = d_head        # Number of heads
        self.d_model = d_k // d_head # Width per individual head slice
        
        # All linear layers operate on the total embedding width (d_k)
        self.W_q = nn.Linear(d_k, d_k, bias=False)
        self.W_k = nn.Linear(d_k, d_k, bias=False)
        self.W_v = nn.Linear(d_k, d_k, bias=False)
        
        self.W_o = nn.Linear(d_k, d_k, bias=False)

    def forward(self, x, mask=None):
        B, T, d_k_dim = x.shape  # x shape: [Batch, Tokens, d_k]
        
        # 1. Unified projection to total highway width
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 2. Slice into your new variable layout: [B, T, d_head, d_model] 
        # Then permute to isolate heads: [B, d_head, T, d_model]
        q = q.view(B, T, self.d_head, self.d_model).permute(0, 2, 1, 3)
        k = k.view(B, T, self.d_head, self.d_model).permute(0, 2, 1, 3)
        v = v.view(B, T, self.d_head, self.d_model).permute(0, 2, 1, 3)
        
        # 3. Calculate Scaled Dot-Product Attention
        # Scaling factor uses your new per-head dimension variable (self.d_model)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # Context matrix shape: [B, d_head, T, d_model]
        context = torch.matmul(attention_weights, v)
        
        # 4. Merge the heads back into sequential layout: [B, T, d_head, d_model]
        context = context.permute(0, 2, 1, 3).contiguous()
        
        # Collapse back into the total embedding width (self.d_k) -> [B, T, d_k]
        concat_output = context.view(B, T, self.d_k)
        
        # 5. Output mix layer
        return self.W_o(concat_output), attention_weights

if __name__ == "__main__":
    # Total dim = 512, Heads = 8 -> Means dimension per head (d_model) will be 64
    mha = MultiHeadAttention(d_k=512, d_head=8)
    sample_tokens = torch.randn(2, 5, 512) # Shape: [Batch=2, Tokens=5, d_k=512]
    
    out, weights = mha(sample_tokens)
    print("Input Tensor Shape:      ", sample_tokens.shape)
    print("Final Output Shape:     ", out.shape)
    print("Attention Weights Shape: ", weights.shape)  # Output: [2, 8, 5, 5]