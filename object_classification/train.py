import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
mpl.use('svg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.transforms import functional as tf

import util
from data import VOC07
from model import Model

img_dir = '/home/jsharp/DL/competition2/yolo_crop/'
voc07 = VOC07(img_dir)

n_sample = len(voc07)

pivot = n_sample * 9 // 10
voc07train = Subset(voc07, range(n_sample)[:pivot])
voc07valid = Subset(voc07, range(n_sample)[pivot:])
# voc07visul = ConcatDataset([
#     Subset(voc07train, random.sample(range(len(voc07train)), 50)),
#     Subset(voc07train, random.sample(range(len(voc07valid)), 50)),
# ])

train_loader = DataLoader(voc07train, batch_size=32, shuffle=True)
valid_loader = DataLoader(voc07valid, batch_size=32, shuffle=False)
# visul_loader = DataLoader(voc07visul, batch_size=32, shuffle=False)

device = 'cuda:1'
model = Model().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)

def train(pbar):
    model.train()
    metrics = {'loss': util.RunningAverage()}
    for img_b, label_b in iter(train_loader):
        img_b = img_b.to(device)
        label_b = label_b.to(device)

        optimizer.zero_grad()
        out_b = model(img_b)
        label_b = torch.squeeze(label_b)
        loss = criterion(out_b, label_b)
        loss.backward()
        optimizer.step()

        metrics['loss'].update(loss.item())
        pbar.set_postfix(metrics)
        pbar.update(len(img_b))
    return metrics


def valid(pbar):
    model.eval()
    metrics = {'loss': util.RunningAverage()}
    correct =  0
    total = n_sample/10
    for img_b, label_b in iter(valid_loader):
        img_b = img_b.to(device)
        label_b = label_b.to(device)

        out_b = model(img_b)
        label_b = torch.squeeze(label_b)
        loss = criterion(out_b, label_b)

        _, p_b = torch.max(out_b, 1)
        correct += (p_b == label_b).sum().item()

        metrics['loss'].update(loss.item())
        pbar.set_postfix(metrics)
        pbar.update(len(img_b))

    print('Accuracy : {}'.format(correct/total))
    
    return {f'val_{k}': v for k, v in metrics.items()}


# def visul(pbar, epoch):
#     model.eval()
#     epoch_dir = log_dir / f'{epoch:03d}'
#     epoch_dir.mkdir(parents=True)
#     for img_b, label_b in iter(visul_loader):
#         out_b = model(img_b.to(device)).cpu()
#         h, w = img_b.size()[-2:]
#         hmp_b = F.interpolate(hmp_b, size=(h, w))
#         hmp_b = hmp_b.unsqueeze(2) # [N, 20, 1, 80, 80]
#         out_b = F.interpolate(out_b, size=(h, w))
#         out_b = out_b.unsqueeze(2) # [N, 20, 1, 80, 80]
#         for img, hmp, out in zip(img_b, hmp_b, out_b):
#             img = tf.to_pil_image(img)
#             hmp = [tf.to_pil_image(mp) for mp in hmp]
#             out = [tf.to_pil_image(ot) for ot in out]
#             img_path = epoch_dir / f'{pbar.n:03d}_img.svg'
#             vis_path = epoch_dir / f'{pbar.n:03d}_vis.svg'
#             svg.save([
#                 svg.pil(img)
#             ], img_path, (h, w))
#             svg.save([
#                 *[svg.pil(mp) for mp in hmp],
#                 *[svg.pil(ot) for ot in out],
#             ], vis_path, (h, w), pad_val='white', per_row=10)
#             pbar.update(1)


def log(train_metrics, valid_metrics, epoch):
    json_path = log_dir / 'log.json'
    if json_path.exists():
        df = pd.read_json(json_path)
    else:
        df = pd.DataFrame()

    metrics = {'epoch': epoch, **train_metrics, **valid_metrics}
    df = df.append(metrics, ignore_index=True)
    df = df.astype('str').astype('float')
    with json_path.open('w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)

    fig, ax = plt.subplots(dpi=100, figsize=(10, 6))
    df[['loss', 'val_loss']].plot(kind='line', ax=ax)
    fig.savefig(log_dir / 'loss.svg')
    plt.close()

    if df['val_loss'].idxmin() == epoch:
        torch.save(model, log_dir / 'model.pkl')


for epoch in range(30):
    scheduler.step()

    print('Epoch', epoch)

    with tqdm(total=len(voc07train), desc='  Train', ascii=True) as pbar:
        train_metrics = train(pbar)

    with torch.no_grad():
        with tqdm(total=len(voc07valid), desc='  Valid', ascii=True) as pbar:
            valid_metrics = valid(pbar)
        # with tqdm(total=len(voc07visul), desc='  Visul', ascii=True) as pbar:
        #     visul(pbar, epoch)
        log(train_metrics, valid_metrics, epoch)
