import torch
from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Router(nn.Module):

    def __init__(self, n_embd, num_experts):
        super().__init__()

        self.router = nn.Linear(n_embd, num_experts,bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        logits = self.router(x)
        # wei = self.dropout(wei)
        return logits