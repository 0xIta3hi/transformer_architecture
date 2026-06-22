import torch.nn as nn
import torch
import torch.nn.functional as F

class FeefForwardNetwork(nn.Module):
    def __init__(self, d_k:int, d_ff:int):

        self.fused_gate_up = nn.Linear(d_k, 2*d_ff, bias=False)

        self.down_proj = nn.Linear(d_k, 2*d_ff, bias=False)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        combined_projections = self.fused_gate_up(x)
        gate_track, up_track = combined_projections.chunk(2, dim=-1)
        gated_hidden_states = F.silu(gate_track) * up_track
        output = self.down_proj(gated_hidden_states)
        return output
    