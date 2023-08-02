import torch
import torch.nn as nn
from tqdm import trange

class NCF(nn.Module):
    def __init__(self, df, K, layers):
        super(NCF, self).__init__()
        self.user_n = df.userId.max() + 1
        self.item_n = df.movieId.max() + 1
        self.K = K
        self.layers = layers

        # MF part
        self.embedding_user_mf = nn.Embedding(num_embeddings=self.user_n, embedding_dim=self.K)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.item_n, embedding_dim=self.K)

        # MLP part
        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.user_n, embedding_dim=self.K)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.item_n, embedding_dim=self.K)

        # Layer
        self.fc_layers = nn.ModuleList()
        for idx, (in_features, out_features) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_features=int(in_features), out_features=int(out_features)))

        # Output
        self.last_layer = nn.Linear(in_features=(int(layers[-1]) + self.K), out_features=1)
        self.output_layer = nn.Sigmoid()
        
    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # GMF
        gmf_layer = torch.mul(user_embedding_mf, item_embedding_mf)

        # MLP
        mlp_layer = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        for idx in range(len(self.fc_layers)):
            mlp_layer = self.fc_layers[idx](mlp_layer)
            mlp_layer = nn.ReLU()(mlp_layer)
        
        # Concatenate
        neu_mf_layer = torch.cat([gmf_layer, mlp_layer], dim=-1)

        # Output
        return self.output_layer(self.last_layer(neu_mf_layer)).view(-1)