import torch
from sklearn.metrics import f1_score, classification_report

from data import init_loaders
from train_model import init_model, BATCH_SIZE, IMG_SIZE

device = torch.device('cpu')
print(f"Device: {device}")

# Load the test data
path = 'data/Garbage classification/Garbage classification'
(_, _, test_dataset, _, _, test_loader) = init_loaders(
        path,
        BATCH_SIZE,
        IMG_SIZE)

# Load the model
model = init_model().to(device)
model.load_state_dict(torch.load('/app/models/model.pt'))

model.eval()

# Get the predictions
val_f1 = 0
full_preds = []
full_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        full_preds.extend(preds.tolist())
        full_labels.extend(labels.tolist())

val_f1 = f1_score(full_labels, full_preds, average='weighted')
print(f"Test F1: {val_f1:4f}")
print(classification_report(full_labels, full_preds))
