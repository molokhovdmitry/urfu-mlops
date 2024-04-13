import numpy as np
import pandas as pd
import datetime
import math


def generate_dataset(noise_multiplier=1, seed=42):
    # Set the seed for reproducibility
    np.random.seed(seed)

    # 7 years of data
    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2024, 12, 31)

    dates = pd.date_range(start_date, end_date, freq='D')

    t_range = 10
    t_mean = 20

    sine_values = np.sin(np.arange(len(dates)) / len(dates) * 7 * 2 * math.pi)
    base_temperatures = sine_values * t_range + t_mean

    # Add random noise
    temperature_values = base_temperatures\
        + noise_multiplier * np.random.normal(
            loc=0,
            scale=0.5,
            size=len(dates)
        )

    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature_values
    })

    print(df.head())

    return df


if __name__ == '__main__':
    # Generate train and test datasets
    noise_multipliers = [0.1, 1, 2]
    seeds = [1, 2, 3]
    train_size = 0.8

    for noise_multiplier, seed in zip(noise_multipliers, seeds):
        df = generate_dataset(noise_multiplier, seed)
        train_rows = int(len(df) * train_size)
        df_train = df[:train_rows]
        df_test = df[train_rows:]
        df_train.to_csv(f'train/df_train_{seed}.csv', index=False)
        df_test.to_csv(f'test/df_test_{seed}.csv', index=False)
