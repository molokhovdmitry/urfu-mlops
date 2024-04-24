import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from train_model import SimpleCNN, BATCH_SIZE, IMG_SIZE
from data import init_loaders

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('model.pt'))

train_df = pd.read_csv('data/train.csv', index_col=0)
test_df = pd.read_csv('data/valid.csv', index_col=0)

# Set the model to evaluation mode
model.eval()

_, _, test_loader = init_loaders(BATCH_SIZE, IMG_SIZE)

# Get the predictions
with torch.no_grad():
    predictions = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get class predictions
        _, predicted = torch.max(probabilities, 1)
        predictions.extend(predicted.tolist())

print(
    "Accuracy score:",
    '{:.5f}'.format(accuracy_score(test_df.label, predictions)),
)
print(
    "Accuracy score on random predictions:",
    '{:.5f}'.format(accuracy_score(test_df.label,
                                   np.random.randint(0, 2, size=3000))),
)
