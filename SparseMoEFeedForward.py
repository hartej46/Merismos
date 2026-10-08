import torch
from torch import nn
from model_code.router import Router
from model_code.experts import expert
from deviceConfig import device

class sparseMoEFeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.router_1 = Router(n_embd=768, num_experts=4)
        self.experts = nn.ModuleList([expert(n_embd=768) for _ in range(4)])

    def forward(self, x):

        B, T, C = x.shape

        x = x.reshape(B*T, C) # We dont need different sentences so we combine them all

        logits = self.router_1(x) # (B*T, num_experts)

        logits = torch.softmax(logits, dim=-1) # (B*T, num_experts)

        highest_expert_indices = torch.argmax(logits, dim=-1) # (B*T,)
        highest_expert_value = torch.max(logits, dim=-1).values # (B*T,)

        final_output = torch.zeros_like(x) # (B*T, C)

        capacity_factor = 1.25
        num_experts = len(self.experts)
        capacity = int((B * T / num_experts) * capacity_factor)



        for i, current_expert in enumerate(self.experts):

            mask = (highest_expert_indices == i)

            idx = mask.nonzero(as_tuple = False).squeeze(-1)
            num_token = idx.shape[0]
            print(idx)
            # Extract only the specific tokens for this expert
            tokens_for_expert = x[idx]

            if num_token > capacity:
                tokens_for_expert = highest_expert_value[idx]
                
                values, top_idx = torch.topk(tokens_for_expert, capacity)

                idx = idx[top_idx]
                
            # If no tokens chose this expert, skip to the next one
            if tokens_for_expert.shape[0] == 0:
                continue
                
            # Process the tokens
            expert_out = current_expert(tokens_for_expert)
            
            # Multiply by their specific gate values
            gates = highest_expert_value[idx].unsqueeze(-1)
            weighted_out = expert_out * gates
            
            # Scatter them back into the blank canvas
            final_output[idx] = weighted_out

        return final_output.reshape(B, T, C)


if __name__ == "__main__":
    dummy_x = torch.randn(2,8,768).to(device)
    model = sparseMoEFeedForward().to(device = device)

    print(model(dummy_x).shape)