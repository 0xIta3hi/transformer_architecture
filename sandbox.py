import torch
import torch.nn.functional as F

x = torch.randn(1,3,4)
print("Input Shape (x):", x.shape)

W_q = torch.randn(4,4)
W_k = torch.rand(4,4)
W_v = torch.randn(4,4)

Q = x @ W_q
K = x @ W_k
V = x @ W_v

print("Q Shape:", Q.shape)

scores = Q @ K.permute(0,2,1)
d_k = Q.shape[-1]
attention_weights = F.softmax(scores / (d_k ** 0.5), dim=-1)
print("attention Weight Matrix:\n", attention_weights)

output = attention_weights @ V
print("final Output shape: ", output.shape)

