import torch
import torchvision.utils as u
import torchvision.transforms as transforms
from dataloader_dtd import class_loaders
from dataloader_gaussian import gaussian_loader
import torch.nn as nn
import torch.optim as optim
import os
import gc
import argparse
import torch
import os
import torch 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from PIL import Image

OMP_NUM_THREADS=1

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import (
    VGG16_representations,
    VGG19_representations, 
    alexnet_representations, 
    resnet18_representations, 
    resnet34_representations, 
    resnet50_representations,
    resnet101_representations,
    resnet152_representations,
    inceptionv3_representations,
    densenet121_representations,
    densenet169_representations,
    densenet201_representations,
    )

model_dict = {
    "vgg16": VGG16_representations,
    "vgg19": VGG19_representations,
    "alexnet": alexnet_representations,
    "resnet18": resnet18_representations,
    "resnet34": resnet34_representations,
    "resnet50": resnet50_representations,
    "resnet101": resnet101_representations,
    "resnet152": resnet152_representations,
    "inceptionv3": inceptionv3_representations,
    "densenet121": densenet121_representations,
    "densenet169": densenet169_representations,
    "densenet201": densenet201_representations,
}

texture_choices = ["blotchy", "matted", "pebble", "scaly", "striped"]

#parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=model_dict.keys(), 
                    help="Which model to run")
parser.add_argument("--texture", type=str, required=True, choices=texture_choices, 
                    help="Which texture to generate")

args = parser.parse_args()
model_name = args.model
texture_name = args.texture
model = model_dict[args.model]()

MSE = torch.nn.MSELoss()
device = "cuda"
optim_steps = 5999 
#1 if in orig mode; 30000 if in reco mode; 160000 for best generation quality
mode = "reco"
subset = ""

#reconstruct the texture
text = texture_name #reconstruction of the textures displayed in the paper only
reco_path = f"/y0ur/path/{text}_reco/"

for d in [reco_path]: 
    os.makedirs(d, exist_ok=True)

batch_size = 1

img_path = f"/your/path/gram_matrices_analyses/{text}"
img_dir = os.listdir(img_path)
img_list = []
for f in img_dir:
    img_list.append(Image.open(os.path.join(img_path, f))) 

#params for image transformations
resize = 224
image_index = 0

class ImgDataset(Dataset):
    def __init__(self, img_list, resize=resize):
        self.img_list = img_list

        #add a preload transformation variable that contains the heaviest(?) transformations
        self.transform = Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5137, 0.4639, 0.4261), (0.2576, 0.2330, 0.2412)),
        ])

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img = self.img_list[index]
        if isinstance(img, Image.Image):
            return self.transform(img)
        else:
            raise ValueError("Expected a PIL.Image object.")

dataset = ImgDataset(img_list, resize=resize)
loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1)

num_images = 1
image_size = (3, 224, 224)
mean = 0.5
std = 0.2
#values of gaussian distrib centered around dtd normalizaion values

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
    
gaussian_dataset = GaussianImageDataset()
gaussian_loader = DataLoader(gaussian_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

model.zero_grad()
model.eval()
for param in model.parameters():
    param.requires_grad_(False)

#parameters for images
mean = (0.5137, 0.4639, 0.4261)
std = (0.2576, 0.2330, 0.2412)

#parameters for dataloader
batch_size = 1

def denormalize(input, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(device)
    denorm = input*std+mean

    return denorm 

preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

target_image = next(iter(loader)).to(device)
with torch.no_grad():
    orig_gram_matrices, feature_map_m_list, _ = model(target_image)

for orig_image, reco_image in zip(loader, gaussian_loader):

    orig_image = orig_image.to(device)
    reco_image = nn.Parameter(reco_image.clone().detach().to(device))
    optimizer = optim.LBFGS([reco_image], 
                            lr=1, 
                            max_iter=20,
                            line_search_fn="strong_wolfe",
                            history_size=400,
                            tolerance_grad=1e-14,
                            tolerance_change=1e-16,
                            )

    for step in range(optim_steps):

        def closure():
            optimizer.zero_grad()

            reco_gram_matrices, _, _ = model(reco_image)

            sum_gram_matrix_loss = 0
            for orig_g, reco_g, m in zip(orig_gram_matrices, reco_gram_matrices, feature_map_m_list):
                sum_gram_matrix_loss += MSE(orig_g, reco_g) / (4 * m)

            sum_gram_matrix_loss.backward()
            return sum_gram_matrix_loss
        
        sum_gram_matrix_loss = optimizer.step(closure)

        if step % 1000 == 0:
            print(f"step: {step}, gram loss: {sum_gram_matrix_loss.item()}")

    with torch.no_grad():
        denorm_reco = denormalize(reco_image, mean, std).clamp(0, 1)
    u.save_image(denorm_reco, os.path.join(reco_path, f"{model_name}_{text}_reco_s{step}.png"))

torch.cuda.empty_cache()
gc.collect()