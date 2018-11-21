import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.transforms import functional as tf

import util
from data import VOC07
from model import Model

device = 'cuda:1'

img_dir = '/home/jsharp/DL/competition2/yolo_crop/'
voc07 = VOC07(img_dir)
n_sample = 21000

# voc07valid = Subset(voc07, range(n_sample)[pivot:])
valid_loader = DataLoader(voc07, batch_size=32, shuffle=False)

model = torch.load('/home/jsharp/DL/competition2/log/2018.11.17-20:50:49/model.pkl').to(device)

correct =  0
total = n_sample

for img_b, label_b in iter(valid_loader):
        img_b = img_b.to(device)
        label_b = label_b.to(device)

        out_b = model(img_b)
        _, p_b = torch.max(out_b, 1)

        label_b = torch.squeeze(label_b)

        correct += (p_b == label_b).sum().item()

print('Accuracy : {}'.format(correct/total))

