import pandas as pd
import numpy as np
import pytest
from joblib import load
from sklearn.metrics import root_mean_squared_log_error

THRESHOLD_RMSLE = 0.9


def df_to_X_y(df):
    coord_cols = [
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude'
    ]
    X = df.drop('trip_duration', axis=1)[coord_cols]
    y = df['trip_duration']
    return X, y


def score_predictions(df, pipeline):
    X, y_true = df_to_X_y(df)
    y_pred = np.abs(pipeline.predict(X))
    return root_mean_squared_log_error(y_true, y_pred)


@pytest.fixture
def init_pipeline():
    global linear_pipe
    linear_pipe = load('models/linear_pipe.joblib')


def test_mse_on_df2(init_pipeline):
    df = pd.read_csv('data/processed/df2.csv')
    assert score_predictions(df, linear_pipe) < THRESHOLD_RMSLE


def test_mse_on_df3(init_pipeline):
    df = pd.read_csv('data/processed/df3.csv')
    assert score_predictions(df, linear_pipe) < THRESHOLD_RMSLE


def test_mse_on_df_noise(init_pipeline):
    df = pd.read_csv('data/processed/df_noise.csv')
    assert score_predictions(df, linear_pipe) < THRESHOLD_RMSLE
