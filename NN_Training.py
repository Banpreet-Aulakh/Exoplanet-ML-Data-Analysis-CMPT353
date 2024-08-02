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
import joblib

class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, linear_size=48):
        super(LSTMModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, linear_size)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(linear_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
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
        # Get the padded data and labels
        padded_data = self.padded_dfs[idx]
        label = self.labels[idx]

        return padded_data, label

class TransformerModel(nn.Module):
    def __init__(self,num_classes, input_size, hidden_size, num_layers, num_heads, linear_size=6):
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
        self.fc = nn.Linear(input_size, linear_size)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(linear_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def read_data(confirmed_path, fp_path):
    # Get all the files
    confirmed_files = glob.glob(os.path.join(confirmed_path, "*.csv"))
    false_files = glob.glob(os.path.join(fp_path, "*.csv"))

    # Get min length of the two file lists
    filecount = min(len(confirmed_files), len(false_files))

    # convert the files to dataframes
    confirmed_dfs = [(pd.read_csv(file).iloc[:, :13]).drop(columns=['QUALITY', 'ORBITID']) for file in confirmed_files[:filecount]]
    false_dfs = [(pd.read_csv(file).iloc[:, :13]).drop(columns=['QUALITY', 'ORBITID']) for file in false_files[:filecount]]

    return confirmed_dfs, false_dfs

def get_data_loaders(confirmed_dfs, false_dfs, batch_size, test_size=0.08):
    # Scale the data and convert to tensors
    tensors = []
    scaler = StandardScaler()
    for df in confirmed_dfs + false_dfs:
        array = df.to_numpy()
        array = np.where(np.isinf(array), np.nan, array)
        array = np.nan_to_num(array, nan=np.nanmean(array))
        array = scaler.fit_transform(array)
        tensor = torch.tensor(array.astype(np.float32))
        tensors.append(tensor)

    # Pad the tensors
    if batch_size != 1:
        tensors = pad_sequence(tensors, batch_first=True)

    # Create the labels
    labels = torch.tensor([0] * len(confirmed_dfs) + [1] * len(false_dfs))
    labels_onehot = torch.zeros(labels.size(0), 2).scatter_(1, labels.unsqueeze(1), 1)


    # train test split
    tensors_train, tensors_test, labels_train, labels_test = train_test_split(tensors, labels_onehot, test_size=test_size)

    # Create the dataset
    train_dataset = MyDataset(tensors_train, labels_train)
    test_dataset = MyDataset(tensors_test, labels_test)


    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=100, lr=0.001):
    # Set the device for the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using Device: device")
    model.to(device)

    # Clear the cache of gpu if used
    if device == "cuda:0":
        torch.cuda.empty_cache()

    # Set the loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0
        for x, y in train_loader:
            # Move the data to the device
            x = x.to(device)
            y = y.to(device)
            # Get the prediction and calculate the loss
            y_pred = model(x)
            y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_batches += 1
            # Training steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / total_batches
        model.eval()
        # Validation pt 1
        with torch.no_grad():
            total_correct = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
                _, predicted = torch.max(y_pred, dim=1)
                _, true_labels = torch.max(y, dim=1)
                total_correct += (predicted == true_labels).sum().item()
        acc = total_correct / len(train_loader.dataset)
        print(f"End of {epoch}")
        print(f"Training: accuracy {acc}, loss {avg_loss}")
        # Validation pt 2
        with torch.no_grad():
            total_correct = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
                _, predicted = torch.max(y_pred, dim=1)
                _, true_labels = torch.max(y, dim=1)
                total_correct += (predicted == true_labels).sum().item()
            acc = total_correct / len(test_loader.dataset)
            print(f"Validation: accuracy {acc}")
        model.train()

def main():
    confirmed_path = "aws_zipped/confirmed/confirmed_lightcurve_processed/confirmed_lightcurves"
    fp_path = "aws_zipped/false/false_lightcurves"
    print("Reading data")
    conf_dfs, fp_dfs = read_data(confirmed_path, fp_path)

    # LSTM Parameters
    num_classes = 2  # Keep at 2
    input_size = 11  # Keep at 11
    hidden_size = 96
    num_layers = 1 # Must be divisible by input_size so 1 or 11
    num_epochs = 1
    lr = 0.0001
    batch_size = 1 # anything other than 1 and the model will perform significantly worse
    linear_size = 48 # change freely

    # LSTM MODEL
    print("Creating data loaders")
    train_loader, test_loader = get_data_loaders(conf_dfs, fp_dfs, batch_size)
    print("Creating LSTM model")
    model = LSTMModel(num_classes=num_classes, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, linear_size=linear_size)
    print("Training model")
    train_model(model, train_loader, test_loader, num_epochs, lr)
    print("Finished LSTM model")
    joblib.dump(model, "lstm_model.pkl")
    print("Saved LSTM model")


    # Transformer Parameters
    num_classes = 2 # Keep at 2
    input_size = 11  # keep at 11
    hidden_size = 12  # change freely
    num_layers = 1 # Change freely, signficant increase in time
    num_heads = 1 # must be divisible by input_size so 1 or 11
    num_epochs = 100 # Change freely
    lr = 0.0001
    batch_size = 6
    linear_size = 6 # Change freely

    # TRANSFORMER MODEL
    print("Creating data loaders")
    train_loader, test_loader = get_data_loaders(conf_dfs, fp_dfs, batch_size)
    print("Creating TRANSFORMER model")
    model = TransformerModel(num_classes=num_classes, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, linear_size=linear_size)
    print("Training TRANSFORMER model")
    train_model(model, train_loader, test_loader, num_epochs, lr)
    print("Finished TRANSFORMER model")
    joblib.dump(model, "lstm_model.pkl")
    print("Saved LSTM model")



if __name__ == "__main__":
    main()
    
