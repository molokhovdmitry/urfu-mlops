import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def ohe(df: pd.DataFrame) -> pd.DataFrame:
    # Define the transformer
    ohe_cols = ['Source', 'State']
    ohe_pipe = Pipeline([('ohe', OneHotEncoder(max_categories=10))])

    ohe_preprocessor = ColumnTransformer(
        transformers=[
            ('ohe_pipe', ohe_pipe, ohe_cols)
        ], remainder='passthrough'
    )

    # Transform
    print("One hot encoding.")
    df = pd.DataFrame(ohe_preprocessor.fit_transform(df))
    feature_names = ohe_preprocessor.get_feature_names_out()
    feature_names = [feature.split('__')[1] for feature in feature_names]
    df.columns = feature_names

    return df


if __name__ == "__main__":
    path = 'data/US_Accidents_March23.csv'
    print("Loading the data.")
    df = pd.read_csv(path)
    df = ohe(df)
    df.to_csv(path, index=False)
