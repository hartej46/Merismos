import torch
from torch.utils.data import DataLoader
from torch import nn

def train_loop(model : nn.Module, 
               data_loader : DataLoader,
               optimizer : torch.optim.Optimizer = None,
               device : torch.device = torch.device("cpu"),
               epochs : int = 5,
               ):

    model.to(device)
    epoch_loss = []

    for epoch in range(epochs):

        model.train()
        training_loss = 0.0

        for step,(X,y) in enumerate(data_loader):

            X = X.to(device)
            y = y.to(device)

            logits,loss = model(X,y)

            # Gradient clipping is recommended for Transformer/MoE stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(data_loader)}] | Loss: {loss.item():.4f}")

        training_loss /= len(data_loader)
        epoch_loss.append(training_loss)

    return epoch_loss

    