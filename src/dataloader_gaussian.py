import torch
import torch 
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

subset_size = 5
batch_size = 8
num_images = 5640
image_size = (3, 224, 224)
#values of gaussian distrib centered around data normalization values

mean = 0.5
std = 0.2

class GaussianImageDataset(Dataset):
    def __init__(self, num_images=num_images, image_size=image_size, mean=mean, std=std):
        self.num_images = num_images
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        gaussian_img = torch.randn(self.image_size) * self.std + self.mean
        return gaussian_img
    
subset_size = subset_size

gaussian_dataset = GaussianImageDataset()
gaussian_loader = DataLoader(gaussian_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
subset_indices = list(range(len(gaussian_dataset)))[:subset_size]
gaussian_subset = Subset(gaussian_dataset, subset_indices)
gaussian_subset_loaders = DataLoader(gaussian_subset, batch_size=batch_size, shuffle=True)    