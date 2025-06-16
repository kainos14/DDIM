import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def infer_ddim_anomaly(model, test_loader, tau, device='cuda'):
    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            x, label = batch[0].to(device), batch[1].cpu().numpy()
            x_input = x.view(x.shape[0], model.in_channels, -1)
            t = torch.randint(0, model.num_timesteps, (x_input.shape[0],), device=device).long()

            reconstructed = model(x_input, t)
            loss = F.mse_loss(reconstructed, x_input, reduction='none')
            loss = loss.view(loss.shape[0], -1).mean(dim=1).cpu().numpy()  # per-sample loss

            pred = (loss > tau).astype(int) 
            all_preds.extend(pred)
            all_labels.extend(label)
            all_losses.extend(loss)

    return np.array(all_labels), np.array(all_preds), np.array(all_losses)

labels, preds, losses = infer_ddim_anomaly(ddim_model, test_loader, tau, device)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(labels, preds))
print(classification_report(labels, preds, digits=4))
