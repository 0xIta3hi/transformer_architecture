import torch
import torch.nn.functional as F

def scaled_dot_product(Q,K,V,mask=None):
    pass

if __name__ == "__main__":
    test_Q = torch.randn(1, 2, 3, 4) # Batch=1, Heads=2, Tokens=3, d_k=4
    out, weights = scaled_dot_product(test_Q, test_Q, test_Q)
    print("Output Shape:", out.shape)