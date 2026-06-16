import torch
import torch.nn.functional as F

# 1. Simulate a tiny batch: 1 sentence, 3 words, each word has a 4-dimensional vector
# Shape: [Batch Size, Sequence Length, Embedding Dimension]
x = torch.randn(1, 3, 4)
print("Input Shape (x):", x.shape)

# 2. Create raw linear projections for Q, K, and V
# In the paper, these are torch.nn.Linear, but let's use raw matrices for first principles
W_q = torch.randn(4, 4)
W_k = torch.randn(4, 4)
W_v = torch.randn(4, 4)

# 3. Project the input to get your Q, K, and V tensors
Q = x @ W_q
K = x @ W_k
V = x @ W_v
print("Q Shape:", Q.shape) 

# 4. Step 1 of the formula: Calculate Raw Attention Scores (Q multiplied by K Transposed)
# We permute K's last two dimensions to change its shape from [1, 3, 4] to [1, 4, 3]
scores = Q @ K.permute(0, 2, 1)
print("Scores Shape (Q @ K_T):", scores.shape)  

# 5. Step 2 & 3 of the formula: Scale and apply Softmax to get the weights
d_k = Q.shape[-1]
attention_weights = F.softmax(scores / (d_k ** 0.5), dim=-1)
print("Attention Weights Matrix:\n", attention_weights)

# 6. Step 4 of the formula: Multiply weights by Values to get context vectors
output = attention_weights @ V
print("Final Output Shape:", output.shape)  