import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
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


class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        width, height = img.size
        boxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)

        # Normalize box coordinates
        for box_idx, box in enumerate(boxes):
            target['boxes'][box_idx][0] /= (width + 1e-9)
            target['boxes'][box_idx][1] /= (height + 1e-9)
            target['boxes'][box_idx][2] /= (width + 1e-9)
            target['boxes'][box_idx][3] /= (height + 1e-9)

        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([img_id])

        if self.transform:
            img = self.transform(img)
            
        return img, target

    def __len__(self):
        return len(self.ids)
    

class COCODetectionDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, ann_file, batch_size=8, num_workers=0, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        self.full_dataset = COCODetectionDataset(self.img_dir, self.ann_file, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)
"""   

class COCODetectionDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, ann_file, batch_size=8, num_workers=0, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        # Load the full dataset
        self.full_dataset = COCODetectionDataset(self.img_dir, self.ann_file, transform=self.transform)
        
        # Calculate lengths for each split
        dataset_length = len(self.full_dataset)
        train_length = int(0.7 * dataset_length)
        val_length = int(0.2 * dataset_length)
        test_length = dataset_length - train_length - val_length  # Ensure the lengths sum up correctly

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_length, val_length, test_length]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def predict_dataloader(self):
        # For predict, using the full dataset can be an option, but if you need it to be specific you can choose train/val/test
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)
        """
