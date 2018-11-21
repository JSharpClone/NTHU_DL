import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf

class VOC07:
    def __init__(self, img_dir):
        self.img_dir = Path(img_dir).expanduser().resolve()
        self.data = []
        num_c = [1744, 2564, 2268, 1402, 2310, 2056, 6214, 3044, 3482, 1382, 
                2232, 4008, 2268, 2690, 10906, 1690, 1252, 2496, 2006, 1732, 2436]
   
        for i in range(21):
            c = np.random.permutation(num_c[i])[:1000]
            self.data.append(c)

    def __len__(self):
        return 21 * 1000

    def __getitem__(self, idx):
        i = idx // 21
        c = idx % 21
        
        img_path = self.img_dir / str(c) / (str(self.data[c][i])+'.jpg')
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img = tf.to_tensor(img)
        label = torch.LongTensor([c])
        return img, label


if __name__ == '__main__':

    img_dir = '/home/jsharp/DL/competition2/yolo_crop/'
    voc07 = VOC07(img_dir)
    img, label = voc07[20033]
    print(label)
    img.show()

