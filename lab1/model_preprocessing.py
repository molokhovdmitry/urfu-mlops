import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DateFeatures(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        date = pd.to_datetime(X['date'])
        X.insert(0, 'day', value=date.dt.day)
        X.insert(0, 'month', value=date.dt.month)
        X.drop('date', axis=1, inplace=True)

        return X


date_pipe = Pipeline([
    ('date_features', DateFeatures()),
    ('norm', MinMaxScaler())
])

temperature_pipe = Pipeline([
    ('norm', MinMaxScaler()),
])

preprocessors = ColumnTransformer(
    transformers=[
        ('date_features', date_pipe, ['date']),
        ('temperature', temperature_pipe, ['temperature']),
    ], remainder='drop'
)


if __name__ == '__main__':
    df = pd.read_csv('test/df_test_1.csv')

    df_transformed = preprocessors.fit_transform(df)

    print(df_transformed)
