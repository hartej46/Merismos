import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SparseMoEFeedForward import sparseMoEFeedForward
from deviceConfig import device
import torch
from torch import nn

n_embd = 384
block_size = 256
dropout = 0.2
vocab_size = 1115394
alpha = 0.01 # Weight for the load balancing loss

# Same as baseline GPT model
class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

# Same as baseline GPT model
class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# Here we use the SparseMoEFeedForward class instead of the standard FeedForward class
class Block_MoE(nn.Module):

    def __init__(self,n_embd):
        super().__init__()

        self.sa = MultiHeadAttention(num_heads=4, head_size=n_embd//4)
        self.ffwd = sparseMoEFeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        ffn,loss = self.ffwd(self.ln2(x))
        x = x + ffn

        return x,loss

class MoEGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block_MoE(n_embd) for _ in range(4)])
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
    
            B, T = idx.shape
    
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
    
            x = tok_emb + pos_emb
            x,loss = self.blocks(x)
    
            logits = self.lm_head(x)
    
            if targets is None:
                loss = None
    
            else: 
    
                B, T, C = logits.shape
    
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss_cross_entropy= nn.functional.cross_entropy(logits, targets)

                total_loss = loss_cross_entropy + loss*alpha
    
            return logits, total_loss

if __name__ == "__main__":
    # dummy_x = torch.randn(2,8,768).to(device)
    model = sparseMoEFeedForward().to(device = device)

    # print(model(dummy_x).shape)

    from pathlib import Path

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents = True,exist_ok = True)

    MODEL_NAME = "MoE_model"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj = model.state_dict(),f =MODEL_SAVE_PATH)