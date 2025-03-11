import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the image directory.
            transform (callable, optional): A function/transform to apply to the image data.
        """
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg') or filename.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transformations (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)

        return image

def get_dataloader(image_dir = '/imdata',
     batch_size=32,
     num_workers=4,
     transform=None):
    dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

