import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm

def train_loop(model : nn.Module, 
               train_data_loader : DataLoader,
               validation_data_loader : DataLoader,
               optimizer : torch.optim.Optimizer = None,
               device : torch.device = torch.device("cpu"),
               epochs : int = 5,
               ):

    model.to(device)
    epoch_loss = []
    val_loss = []

    for epoch in range(epochs):

        model.train()
        training_loss = 0.0
        validation_loss = 0.0

        for X,y in tqdm(train_data_loader):

            X = X.to(device)
            y = y.to(device)

            _,loss = model(X,y)

            training_loss += loss

            # Gradient clipping is recommended for Transformer/MoE stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        model.eval()
        with torch.inference_mode():

            for X_val, y_val in tqdm(validation_data_loader):

                X_val = X_val.to(device)
                y_val = y_val.to(device)

                _,val_loss = model(X_val,y_val)

                validation_loss += val_loss

        training_loss /= len(train_data_loader)
        validation_loss /= len(validation_data_loader)

        print(f"Epoch [{epoch+1}/{epochs}] |  Loss: {training_loss.item():.4f} | Validation Loss: {validation_loss:.4f}")
        epoch_loss.append(training_loss)
        val_loss.append(validation_loss)

    return epoch_loss,val_loss

    