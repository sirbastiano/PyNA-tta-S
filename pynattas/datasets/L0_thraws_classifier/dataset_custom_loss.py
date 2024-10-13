import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image
import pickle
import os


class SentinelDataset(Dataset):
    def __init__(self, root_dir, split='TrainVal', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (THRAWS).
            split (string): One of ['TrainVal', 'Test'] to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Getting the list of image paths and corresponding labels
        self.image_paths = []
        self.labels = []

        # Load the paths and labels
        for label, event in enumerate(['notevent', 'event']):
            dir_path = os.path.join(self.root_dir, split, event)
            for filename in os.listdir(dir_path):
                if filename.endswith('.pkl'):
                    self.image_paths.append(os.path.join(dir_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with open(self.image_paths[idx], 'rb') as file:
            image = pickle.load(file)
            image = Image.fromarray(image.astype('uint8'), 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class SentinelDataModule(LightningDataModule):
    def __init__(self, root_dir, csv_file=None, batch_size=8, num_workers=0, transform=transforms.ToTensor()):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.sentinel_trainval = SentinelDataset(
                root_dir=self.root_dir,
                split='TrainVal',
                transform=self.transform,
            )
            # Compute weights for balanced sampling
            self.class_weights = make_weights_for_balanced_classes(self.sentinel_trainval.labels, 2)
            print(f"DataModule class_weights set: {self.class_weights}")

            self.sentinel_test = SentinelDataset(
                root_dir=self.root_dir,
                split='Test',
                transform=self.transform,
            )

    def train_dataloader(self):
        #sampler = WeightedRandomSampler(self.weights, len(self.weights))
        return DataLoader(
            dataset=self.sentinel_trainval,
            batch_size=self.batch_size,
            #sampler=sampler,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.sentinel_trainval,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.sentinel_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        pass


def make_weights_for_balanced_classes(labels, nclasses):
    count = [0] * nclasses
    for label in labels:
        count[label] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    return torch.tensor(weight_per_class, dtype=torch.float)

