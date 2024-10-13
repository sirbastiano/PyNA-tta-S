import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl


def custom_collate_fn(batch):
    images, targets = zip(*batch)
    
    # Stack images into a single tensor
    images = torch.stack(images)
    
    # Collect all bounding boxes and labels
    bboxes = [torch.tensor(item['bbox']) for sublist in targets for item in sublist]
    labels = [item['category_id'] for sublist in targets for item in sublist]
    
    # Pad bounding boxes to the same length
    max_len = max(len(t) for t in targets)
    padded_bboxes = []
    padded_labels = []
    
    for target in targets:
        num_boxes = len(target)
        bboxes = torch.tensor([t['bbox'] for t in target])
        labels = torch.tensor([t['category_id'] for t in target])
        
        # Pad bboxes and labels if necessary
        if num_boxes < max_len:
            padding = max_len - num_boxes
            bboxes = torch.nn.functional.pad(bboxes, (0, 0, 0, padding))
            labels = torch.nn.functional.pad(labels, (0, padding), value=-1)
        
        padded_bboxes.append(bboxes)
        padded_labels.append(labels)
    
    # Stack the padded bboxes and labels
    padded_bboxes = torch.stack(padded_bboxes)
    padded_labels = torch.stack(padded_labels)
    
    return images, {'bboxes': padded_bboxes, 'labels': padded_labels}


class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.label_files = [f.replace('.jpg', '.txt') for f in self.img_files]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self.load_labels(label_path)

        #image, boxes = resize_and_normalize(image, boxes, (416,416))
        if self.transform is not None:
            image = self.transform(image)

        target = []
        for i, box in enumerate(boxes):
            target.append({'bbox': box, 'category_id': labels[i]})

        return image, target

    def load_labels(self, label_path):
        boxes = []
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                labels.append(int(parts[0]))
                box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                boxes.append(box)
                
        return boxes, labels


class COCODetectionDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, label_dir, batch_size=8, num_workers=4, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        self.full_dataset = COCODetectionDataset(self.img_dir, self.label_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

