import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def train_ddim_model(model, train_loader, test_loader, num_epochs=20, learning_rate=1e-4, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x = batch[0].to(device)
            x = x.view(x.shape[0], unet_config["params"]["in_channels"], -1)
            t = torch.randint(0, model.num_timesteps, (x.shape[0],), device=device).long()
            optimizer.zero_grad()
            loss = loss_fn(model(x, t), x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Avg Train Loss = {avg_loss:.4f}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


EPOCHES = 1000
train_ddim_model(ddim_model, train_loader, test_loader, num_epochs=EPOCHES, learning_rate=1e-4, device=device)
