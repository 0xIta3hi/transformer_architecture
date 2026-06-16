import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Implements Section 3.3 of the paper: FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        # First transformation projects data up to a higher dimension (d_ff)
        self.w_1 = nn.Linear(d_model, d_ff)
        # Second transformation projects data back down to standard dimension (d_model)
        self.w_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # Apply linear 1 -> ReLU activation -> linear 2
        return self.w_2(torch.relu(self.w_1(x)))

if __name__ == "__main__":
    ffn = PositionWiseFeedForward(d_model=512, d_ff=2048)
    test_input = torch.randn(1, 10, 512)
    test_output = ffn(test_input)
    print("FFN Output Shape:", test_output.shape) # Should be [1, 10, 512]