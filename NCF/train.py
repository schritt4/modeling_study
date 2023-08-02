import argparse
from pprint import pprint
from util import *
from dataset import NCF_Dataset
from model import SGD, NCF

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--layers", nargs="+", default=None, help="feature size of MLP in NCF")
    p.add_argument("--k", type=int, help="latent factor size")
    p.add_argument("--n_epochs", type=int, default=200, help="num of Iterations")
    p.add_argument("--lr", type=float, default=0.01, help="learning rate")
    p.add_argument("--batch_size", type=int, default=None, help="batch size of NCF")
    p.add_argument("--beta", type=float, default=None, help="regularization parameter")
    p.add_argument("--sgd", action="store_true", help="Use SGD")
    p.add_argument("--ncf", action="store_true", help="Use NCF")

    return p.parse_args()

def main(config):
    pprint(vars(config))
    ratings_df = get_movielens("ratings.csv")
    print("Rating set shape:", ratings_df.shape)
    if config.sgd:
        sparse_matrix, test_set = make_sparse_matrix(ratings_df)
        print("Sparse Matrix shape:", sparse_matrix.shape)
        print("Test set length:", len(test_set))
        trainer = SGD(sparse_matrix, config.k, config.lr, config.beta, config.n_epochs)
        trainer.train()
        print("train RMSE:", trainer.evaluate())
        print("test RMSE:", trainer.test_evaluate(test_set))
    elif config.ncf:
        model = NCF(ratings_df, config.k, config.layers)

        # Define loss fuction & Optimizer
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        for epoch in range(config.n_epochs):
            model.train() # Start training

            # Split training & test data
            train_data, test_data = train_test_split(ratings_df)
            train_loader = DataLoader(NCF_Dataset(train_data), batch_size=config.batch_size)
            test_loader = DataLoader(NCF_Dataset(test_data), batch_size=config.batch_size)
            for user, item, rating in train_loader:
                output = model(user, item)
                loss = loss_fn(output, rating)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                print(f"LOSS:{loss.item()}")
    else:
        raise RuntimeError()

if __name__ == "__main__":
    config = define_argparser()
    main(config)