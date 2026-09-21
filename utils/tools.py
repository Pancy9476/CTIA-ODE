import os
import random
import numpy as np
import torch

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_to_file(file_path, text):
    print(text)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std = np.maximum(self.std, 1e-5)

    def transform(self, data):
        return (data - self.mean) / self.std

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    for x, history_time, future_time, y in dataloader:
        x = x.to(device, non_blocking=True)
        history_time = history_time.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        preds = model(x, history_time, future_time)

        total_squared_error += (preds - y).pow(2).sum().item()
        total_absolute_error += (preds - y).abs().sum().item()
        total_elements += y.numel()

    mse = total_squared_error / total_elements
    mae = total_absolute_error / total_elements

    return mse, mae