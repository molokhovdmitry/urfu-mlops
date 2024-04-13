import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model_preprocessing import preprocessors


def make_dataloader(path):
    # Load the data
    data = preprocessors.fit_transform(pd.read_csv(path))

    # Split the data into features (X) and target (y)
    X = data[:, (0, 1)]
    y = data[:, 2]

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class LSTMModel(nn.Module):
    def __init__(
            self,
            input_size=2,
            hidden_size=50,
            num_layers=2,
            output_size=1):

        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, num_epochs, dataloader):

    num_epochs = num_epochs

    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X.unsqueeze(1))
            loss = criterion(outputs, batch_y.unsqueeze(1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    return model


if __name__ == '__main__':
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for dataset in [f'train/df_train_{i+1}.csv' for i in range(3)]:
        dataloader = make_dataloader(dataset)
        model = train_model(model, 20, dataloader)
    torch.save(model.state_dict(), 'model.pt')
