'''
Workshop #3 - Linear Regression & Neural Networks

Welcome again! This is the code used in the third workshop. Here, you will see how
we can use Neural Networks in Deep Learning (Not Machine Learning anymore xD) to make 
extrodinary results.

For that, we will be using the same data from the Linear Regression model!

As always, feel free to play around with this code :)

Use this to Learn and remember to always have fun!
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from linearRegression import df
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

scaler = MinMaxScaler()

# Apply scaler() to all the columns except for the 'yes-no' (1/0)
scale_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df[scale_columns] = scaler.fit_transform(df[scale_columns])

x_train = df.drop(columns= 'price')
y_train = df['price']

# Convert Data to PyTorch Tensors
x_train_t = torch.tensor(x_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

class HousePricePredictor(nn.Module):
    def __init__(self):
        super(HousePricePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_train.shape[1], 256),  # More neurons
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output layer
        )

    def forward(self, x):
        return self.model(x)

# K-Fold Cross Validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
r2_scores = []

best_r2 = -np.inf  # Track best model
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_t)):
    print(f"\n🔄 Fold {fold + 1}/{k_folds}")

    x_train_fold, y_train_fold = x_train_t[train_idx], y_train_t[train_idx]
    x_val_fold, y_val_fold = x_train_t[val_idx], y_train_t[val_idx]

    model = HousePricePredictor()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
    
    best_fold_r2 = -np.inf
    patience = 20  # Early stopping patience
    counter = 0

    for epoch in range(2000):  # Train for more epochs
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train_fold)
        loss = criterion(predictions, y_train_fold)
        loss.backward()
        optimizer.step()

        # Evaluate every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_fold = model(x_val_fold)
            fold_r2 = r2_score(y_val_fold.numpy(), y_pred_fold.numpy().flatten())

            if fold_r2 > best_fold_r2:
                best_fold_r2 = fold_r2
                counter = 0  # Reset patience counter
            else:
                counter += 1

            if counter >= patience:
                print(f"⏹ Early Stopping at Epoch {epoch}")
                break

    print(f"✅ Fold {fold + 1} Best R² Score: {best_fold_r2:.4f}")
    r2_scores.append(best_fold_r2)

    if best_fold_r2 > best_r2:
        best_r2 = best_fold_r2
        best_model = model  # Save best model

mean_r2 = np.mean(r2_scores)
print(f"\n📊 **Final Cross-Validation R² Score: {mean_r2:.4f}**")