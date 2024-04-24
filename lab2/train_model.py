import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import init_loaders


IMG_SIZE = 256
BATCH_SIZE = 128
EPOCHS = 8


# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, 5, padding='same')
        self.fc1 = nn.Linear(8 * (IMG_SIZE//4) * (IMG_SIZE//4), 80)
        self.fc2 = nn.Linear(80, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(device, train_loader, val_loader):
    # Initialize the model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(
            f"Epoch {epoch + 1}",
            "Training Loss:",
            '{:.5f}'.format(running_loss / len(train_loader)),
            "Validation Loss:",
            '{:.5f}'.format(val_loss / len(val_loader))
        )

    # Save the model
    torch.save(model.state_dict(), 'model.pt')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    train_loader, val_loader, _ = init_loaders(BATCH_SIZE, IMG_SIZE)
    train(device, train_loader, val_loader)
