import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf

import util

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    b1_x1 = box1[0]
    b1_y1 = box1[1]
    b1_x2 = box1[2]
    b1_y2 = box1[3]

    b2_x1 = box2[0]
    b2_y1 = box2[1]
    b2_x2 = box2[2]
    b2_y2 = box2[3]
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

class VOC07:
    def __init__(self, ann_path, img_dir, img_size=(320, 320)):
        self.ann_path = Path(ann_path).expanduser().resolve()
        self.img_dir = Path(img_dir).expanduser().resolve()
        self.img_size = img_size
        self.num_pic = np.zeros(21, dtype=int)
        self.bg_count = 0

        with self.ann_path.open() as f:
            data = [line.strip() for line in f.readlines()]

        self.anns = []
        for line in data:
            line = line.split()
            vals = np.int32(line[1:]).reshape(-1, 5)
            self.anns.append({
                'img_name': line[0],
                'n_obj': len(vals),
                'boxs': vals[:, :-1],
                'lbls': vals[:, -1],
            })
        
        for i in range(len(self.anns)):
            self.expand(i)



        

    def __len__(self):
        return len(self.imgs)

    def expand(self, idx):
        ann = self.anns[idx]

        img_path = self.img_dir / ann['img_name']
        img = Image.open(img_path).convert('RGB')
        boxs = torch.FloatTensor(ann['boxs'])
        lbls = ann['lbls']

        srcW, srcH = img.size
        dstH, dstW = self.img_size
        img = img.resize((dstW, dstW))

        boxs = boxs / torch.FloatTensor([srcW, srcH, srcW, srcH])
        boxs = boxs * torch.FloatTensor([dstW, dstH, dstW, dstH])

        augs = torch.rand(40, 4) 
        augs = torch.cat([torch.ones((1, 4)), augs], 0)

        for box in boxs:
            for aug in augs:
                aug_box = box * aug

                if bbox_iou(box, aug_box) > 0.5:
                    label = lbls
                else:
                    self.bg_count += 1
                    label = np.array([20])

                save_path = Path('/home/jsharp/DL/competition2/yolo_crop').expanduser().resolve()
               
                save_path /= str(label[0]) 
                save_path.mkdir(parents=True, exist_ok=True)

                if label[0] != 20 or (label[0] == 20 and self.bg_count%500 == 0):
                    crop_img = img.crop(aug_box.numpy())
                    crop_img = crop_img.resize((dstW, dstW))
                    crop_img.save(save_path/(str(self.num_pic[label][0])+'.jpg'))
                    self.num_pic[label[0]] += 1

                    flip_img = crop_img.transpose(Image.FLIP_LEFT_RIGHT)
                    flip_img.save(save_path/(str(self.num_pic[label][0])+'.jpg'))
                    self.num_pic[label[0]] += 1


                
   
        return 

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


if __name__ == '__main__':

    ann_path = '/home/jsharp/dataset/voc2007/pascal_voc_training_data.txt'
    img_dir = '/home/jsharp/dataset/voc2007/VOCdevkit_train/VOC2007/JPEGImages/'
    voc07 = VOC07(ann_path, img_dir)