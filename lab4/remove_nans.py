import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def remove_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    # Get numerical and categorical column names with NaN values
    nan_mask = df.isna().sum() > 0
    nan_cols = df.columns[nan_mask]
    num_nan_cols = [col for col in nan_cols if df.dtypes[col] in [float, int]]
    cat_nan_cols = [col for col in nan_cols if col not in num_nan_cols]
    assert len(nan_cols) == len(num_nan_cols) + len(cat_nan_cols)

    # Define pipelines for numerical and categorical columns
    num_nan_pipe = Pipeline([('nan', SimpleImputer(strategy='mean'))])
    cat_nan_pipe = Pipeline([('nan', SimpleImputer(strategy='most_frequent'))])

    nan_preprocessor = ColumnTransformer(
        transformers=[
            ('num_nan_pipe', num_nan_pipe, num_nan_cols),
            ('cat_nan_pipe', cat_nan_pipe, cat_nan_cols),
        ], remainder='passthrough'
    )

    # Transform
    print("Removing NaN values.")
    df = pd.DataFrame(nan_preprocessor.fit_transform(df))
    feature_names = nan_preprocessor.get_feature_names_out()
    feature_names = [feature.split('__')[1] for feature in feature_names]
    df.columns = feature_names

    return df


if __name__ == "__main__":
    path = 'data/US_Accidents_March23.csv'
    print("Loading the data.")
    df = pd.read_csv(path)
    df = remove_nan_values(df)
    df.to_csv(path, index=False)
