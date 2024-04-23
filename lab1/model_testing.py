import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from model_preparation import LSTMModel
from model_preprocessing import preprocessors

model = LSTMModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()

for dataset in [f'test/df_test_{i+1}.csv' for i in range(3)]:
    df = pd.read_csv(dataset)
    y = df['temperature']
    data = preprocessors.fit_transform(df)
    X = data[:, (0, 1)]
    X = torch.tensor(X, dtype=torch.float)

    with torch.no_grad():
        predictions = model(X.unsqueeze(1)).numpy().flatten()

    scaler = preprocessors.transformers_[1][1][0]
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    print(f"MSE for {dataset}: {mean_squared_error(predictions, y)}")
