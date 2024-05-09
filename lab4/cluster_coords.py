import pandas as pd
from sklearn.cluster import KMeans


def add_coord_clusters(df: pd.DataFrame, N: int) -> pd.DataFrame:
    # Specify coord columns
    cluster_cols = ['Start_Lat', 'Start_Lng']

    # Fit on a slice of data
    kmeans = KMeans(n_clusters=10)
    df_slice = df.sample(N)
    print("Fitting KMeans on a slice of data.")
    kmeans.fit(df_slice[cluster_cols])

    # Predict
    print("Predicting the clusters.")
    clusters = kmeans.predict(df[cluster_cols])
    df['Coord_Cluster'] = clusters

    return df


if __name__ == "__main__":
    path = 'data/US_Accidents_March23.csv'
    print("Loading the data.")
    df = pd.read_csv(path)
    df = add_coord_clusters(df, 20000)
    df.to_csv(path, index=False)
