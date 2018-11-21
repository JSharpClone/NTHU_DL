import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

import numpy as np


class Model(nn.Module):
    def __init__(self, n_class=21):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 21)
        # print(self.model)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    m = Model()
    x = torch.rand(5, 3, 224, 224)
    p = m(x)
    print(p.size())
    a = torch.empty(5, dtype=torch.long).random_(21)
    print(a.size())
    criterion = nn.CrossEntropyLoss()
    output = criterion(p, a)
    print(output)
    
