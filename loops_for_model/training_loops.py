import torch
from torch.utils.data import DataLoader
from torch import nn
import tqdm

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

        for X,y in tqdm.auto(data_loader):

            X = X.to(device)
            y = y.to(device)

            _,loss = model(X,y)

            # Gradient clipping is recommended for Transformer/MoE stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        training_loss /= len(data_loader)

        print(f"Epoch [{epoch+1}/{epochs}] |  Loss: {training_loss.item():.4f}")
        epoch_loss.append(training_loss)

    return epoch_loss

    