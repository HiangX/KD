import torch.nn as nn

class WeightLayer(nn.Module):
    def __init__(self, input_dim):
        nn.Module.__init__(self)
        self.main = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.main(x)