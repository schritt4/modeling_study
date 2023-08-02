import os
import numpy as np
import pandas as pd
from random import random

MOVIELENS_PATH = "./NCF/datasets"

def get_movielens(file="ratings.csv"):
    df = pd.read_csv(os.path.join(MOVIELENS_PATH, file), encoding="utf-8")
    return df.sample(frac=1).reset_index(drop=True)

def make_sparse_matrix(df):
    sparse_matrix = df.groupby("movieId").apply(lambda x: pd.Series(x["rating"].values, index=x["userId"])).unstack()
    sparse_matrix.index.name = "movieId"

    test_set = []
    idx, jdx = sparse_matrix.fillna(0).to_numpy().nonzero()
    indice = list(zip(idx, jdx))
    np.random.shuffle(indice)

    for i, j in indice[:df.shape[0] // 5]:
        test_set.append((i, j, sparse_matrix.iloc[i, j]))
        sparse_matrix.iloc[i, j] = 0
    return sparse_matrix, test_set