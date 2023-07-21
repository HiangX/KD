import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn as nn
from tqdm import tqdm

class DAPLModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.main = nn.Sequential(
            nn.Linear(5796, 3000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3000, 2000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2000, 200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(200, 1, bias=False)
        )

    def forward(self, x):
        return self.main(x)