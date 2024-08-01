import os
import json
from PIL import Image
import rasterio
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pycocotools.coco import COCO
import torch
import numpy as np


def custom_collate_fn(batch):
    images = []
    targets = []

    for b in batch:
        images.append(b[0])
        
        targets_tmp = {'boxes': torch.tensor(np.array(b[1]['boxes'])),
                       'labels': torch.tensor(np.array(b[1]['labels'])),
                       'image_id': torch.tensor(np.array(b[1]['image_id']))}
        targets.append(targets_tmp)

    # Stack images
    images = torch.stack([torch.tensor(np.array(img)) for img in images], dim=0)


    return images, targets


class SPECTRE_COCO_Dataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Use Rasterio to load the single-channel image
        with rasterio.open(img_path) as src:
            image = src.read()  # Read the single channel
        
        image = image[0,:,:]

        boxes = []
        labels = []
        for ann in anns:
            #xmin = ann['bbox'][0]
            #ymin = ann['bbox'][1]
            #xmax = xmin + ann['bbox'][2]
            #ymax = ymin + ann['bbox'][3]
            #boxes.append([xmin, ymin, xmax, ymax])
            xmin, ymin, w, h = ann['bbox']
            cx = (xmin + w / 2)/512
            cy = (ymin + h / 2)/512
            xmin /= 512
            ymin /= 512
            boxes.append([cx, cy, w, h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])

        if self.transform:
            image = self.transform(image)

        return image, target


class SPECTRE_COCO_DataModule(pl.LightningDataModule):
    def __init__(self, img_dir, ann_file, batch_size=4, num_workers=0, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        dataset = SPECTRE_COCO_Dataset(self.img_dir, self.ann_file, transform=self.transform)
        dataset_size = len(dataset)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)
        test_size = dataset_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)
    
    #@staticmethod
    #def collate_fn(batch):
    #    return tuple(zip(*batch))
    

