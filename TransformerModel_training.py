import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import glob
import torch
from torch import nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, 6)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(6, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, padded_dfs, labels):
        self.padded_dfs = padded_dfs
        self.labels = labels

    def __len__(self):
        return len(self.padded_dfs)

    def __getitem__(self, idx):
        return self.padded_dfs[idx], self.labels[idx]

# Transform Data

# Get file paths
confirmed_path = "aws_zipped/confirmed/confirmed_lightcurve_processed/confirmed_lightcurves"
fp_path = "aws_zipped/false/false_lightcurves"

# Get all the files
confirmed_files = glob.glob(os.path.join(confirmed_path, "*.csv"))
false_files = glob.glob(os.path.join(fp_path, "*.csv"))

# Get min length of the two file lists
filecount = min(len(confirmed_files), len(false_files))

# convert the files to dataframes
confirmed_dfs = [(pd.read_csv(file).iloc[:, :13]).drop(columns=['QUALITY', 'ORBITID']) for file in confirmed_files[:filecount]]
false_dfs = [(pd.read_csv(file).iloc[:, :13]).drop(columns=['QUALITY', 'ORBITID']) for file in false_files[:filecount]]

# Get the max length of the dataframes for padding
max_len = max([df.shape[0] for df in confirmed_dfs + false_dfs])

# Scale the data and convert to tensors
tensors = []
scaler = MinMaxScaler()
for df in confirmed_dfs + false_dfs:
    array = df.to_numpy()
    array = np.where(np.isinf(array), np.nan, array)
    array = np.nan_to_num(array, nan=np.nanmean(array))
    array = scaler.fit_transform(array)
    tensor = torch.tensor(array.astype(np.float32))
    tensors.append(tensor)

# UNCOMMENT THIS LINE TO PAD SEQUENCE AND ALLOW FOR BATCH SIZE LARGER THAN 1
# tensors = pad_sequence(tensors, batch_first=True)

# Split the data into training and validation sets
labels = torch.tensor([0] * len(confirmed_dfs) + [1] * len(false_dfs))
labels_onehot = torch.zeros(labels.size(0), 2).scatter_(1, labels.unsqueeze(1), 1)

# Split into training and validation sets
tensors_train, tensors_val, labels_train, labels_val = train_test_split(tensors, labels_onehot, test_size=0.08)

# Create the training and validation data loaders
train_dataset = MyDataset(tensors_train, labels_train)
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = MyDataset(tensors_val, labels_val)
val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


# Set the device for the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Clear the cache of gpu if used
if device == "cuda:0":
    torch.cuda.empty_cache()

model = TransformerModel(input_size=11, hidden_size=25, num_layers=1, num_heads=1)
model.to(device)
loss_fn = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.0001)

n_epochs = 1

for epoch in range(n_epochs):
    total_loss = 0
    total_batches = 0
    for batch_data, batch_labels in train_data_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        prediction = model(batch_data)
        loss = loss_fn(prediction, batch_labels)
        total_loss += loss.item()
        total_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / total_batches
    model.eval()
    # Validation: Training pt 1
    with torch.no_grad():
        total_correct = 0
        for x, y in train_data_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
            _, predicted = torch.max(y_pred, dim=1)
            _, true_labels = torch.max(y, dim=1)
            total_correct += (predicted == true_labels).sum().item()
    acc = total_correct / len(train_data_loader.dataset)
    print(f"End of {epoch}, accuracy {acc}, loss {avg_loss}")
    # Validation pt 2
    with torch.no_grad():
        total_correct = 0
        for x, y in val_data_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
            _, predicted = torch.max(y_pred, dim=1)
            _, true_labels = torch.max(y, dim=1)
            total_correct += (predicted == true_labels).sum().item()
        acc = total_correct / len(val_data_loader.dataset)
        print(f"Validation accuracy {acc}")
    model.train()
