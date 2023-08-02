import numpy as np

from torch.utils.data import Dataset

class NCF_Dataset(Dataset):
    def __init__(self, df):
        self.users = df["userId"].values
        self.items = df["movieId"].values
        self.ratings = df["rating"].values.astype(np.float32)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]