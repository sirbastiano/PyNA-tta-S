import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule

import pickle
import os
import logging
import numpy as np
from typing import Optional

# Initialize logging
logging.basicConfig(level=logging.INFO)

DEFAULT_BATCH_SIZE = 8 # Default batch size for DataLoader
DEFAULT_NUM_WORKERS = 4 # Number of workers for DataLoader, set to improve performance


class ImageLoader:
    def __init__(self, path: str, permute: bool = False):
        """
        Initializes the ImageLoader with a given path.

        Args:
            path (str): Path to the image file (in pickle format).
        """
        self.path = path
        self.permute = permute # permute dimensions (HWC <-> CHW)
        self.load_image()

    def load_image(self) -> Optional[torch.Tensor]:
        """
        Loads the image from the given path and converts it to a torch tensor.
        If the image is not in the expected numpy format, it returns None.
        
        Returns:
            Optional[torch.Tensor]: The loaded image as a torch tensor, or None if the loading fails.
        """
        if not os.path.exists(self.path):
            logging.error(f"File not found: {self.path}")
            return None
        
        try:
            with open(self.path, 'rb') as file:
                image = pickle.load(file)
                
                # Check if the image is a NumPy array
                if isinstance(image, np.ndarray):
                    # Convert the image to torch tensor and permute dimensions (HWC -> CHW)
                    image_tensor = torch.from_numpy(image).float()
                    if self.permute:
                        image_tensor = image_tensor.permute(2, 0, 1)
                    return image_tensor
                else:
                    logging.error(f"Image is not a numpy array: {self.path}")
                    return None
        except (pickle.UnpicklingError, EOFError) as e:
            logging.error(f"Error loading image from {self.path}: {e}")
            return None


class GenericDataoader(Dataset):
    def __init__(self, root_dir, split='TrainVal', transform=None):
        """
        Generic Dataset class that automatically maps class names to labels by scanning directories.
        
        Args:
            root_dir (str): Root directory with dataset folders.
            split (str): Dataset split to use, e.g., 'TrainVal' or 'Test'. (You folder structure should be root_dir/split/class_name/image.pkl)
            transform (callable, optional): Transform to apply to the images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Automatically load class names (subdirectories) and assign labels
        self.classes = self._load_class_names()
        self.image_paths, self.labels = self._load_images_and_labels()

    def _load_class_names(self):
        """
        Loads the class names from the subdirectories inside the root_dir/split folder.
        
        Returns:
            List[str]: List of class names.
        """
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory '{split_dir}' not found.")
        
        # Dynamically load the class names as subdirectory names
        class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        
        if not class_names:
            raise ValueError(f"No class directories found in '{split_dir}'")
        
        return class_names

    def _load_images_and_labels(self):
        """
        Loads image paths and corresponding labels from the dataset directory.
        
        Returns:
            Tuple[List[str], List[int]]: A tuple of (image_paths, labels).
        """
        image_paths = []
        labels = []
        
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, self.split, class_name)
            
            if not os.path.isdir(class_dir):
                raise ValueError(f"Class directory '{class_dir}' not found.")
            
            # Iterate over files in class directory
            for filename in os.listdir(class_dir):
                if filename.endswith('.pkl'):  # Filter .pkl files
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(label)

        if not image_paths:
            raise ValueError(f"No image files found in '{self.split}' directory.")
        
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the image and label corresponding to the given index.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            Tuple[Image.Image, int]: Tuple containing the image and its label.
        """
        # Load the image from the pickle file
        image_loader = ImageLoader(self.image_paths[idx])
        
        # Apply the optional transformation
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class GenericDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        csv_file: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        transform=None
    ):
        """
        Initializes the data module.

        Args:
            root_dir (str): Root directory containing the data.
            csv_file (Optional[str]): Path to CSV file for any additional dataset handling (if needed).
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            transform (callable): Transformations to apply on the data.
        """
        super().__init__()
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Use this method to download or prepare data. No heavy lifting should happen here."""
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for different stages (fit, test).
        
        Args:
            stage (Optional[str]): Stage to set up ('fit', 'validate', 'test', or 'predict').
        """
        if stage == 'fit' or stage is None:
            # Initialize training and validation datasets
            self.train_dataset = GenericDataset(
                root_dir=self.root_dir,
                split='TrainVal',
                transform=self.transform,
            )
            # You can split into train/val sets here if needed

        if stage == 'test' or stage is None:
            self.test_dataset = GenericDataset(
                root_dir=self.root_dir,
                split='Test',
                transform=self.transform,
            )

    def train_dataloader(self):
        """Returns the training DataLoader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle during training
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Returns the validation DataLoader (can use the same dataset in some cases)."""
        return DataLoader(
            dataset=self.train_dataset,  # Assuming same dataset, but split internally or shuffle=False
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Returns the test DataLoader."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Returns the DataLoader used for prediction."""
        # Use a separate dataset or the same test dataset for predictions
        return DataLoader(
            dataset=self.test_dataset,  # Assuming test dataset for predictions
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )