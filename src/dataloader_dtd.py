import torch
import os
import torch 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose
from PIL import Image

subset_size = 10
batch_size = 8

img_path = "/your/path"
img_dir = os.listdir(img_path)

#prepare images
img_dict = {}
for texture_class in img_dir:
    class_path = os.path.join(img_path, texture_class)
    if os.path.isdir(class_path):
        img_list = []
        for img in os.listdir(class_path):
            file_path = os.path.join(class_path, img)
            try:
                with Image.open(file_path) as image:
                    img_list.append(image.copy())
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        img_dict[texture_class] = img_list

#params for image transformations
resize = 224
image_index = 0

#params for train and test splitting
train_split = 0.8
test_split = 0.2
random_state = 42

class ImgDataset(Dataset):
    def __init__(self, img_list, resize=resize):
        self.img_list = img_list

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
  
subset_size = subset_size

class_loaders = {}
class_subset_loaders = {}

for class_name, img_list in img_dict.items():
    dataset = ImgDataset(img_list, resize=resize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_loaders[class_name] = loader

    subset_indices = list(range(len(dataset)))[:subset_size]
    subset = Subset(dataset, subset_indices)
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)   
    class_subset_loaders[class_name] = subset_loader 


subset_indices = torch.randperm(len(dataset))[:subset_size]
subset = Subset(dataset, subset_indices)

#uncomment and run the following to get custom normalization parameters (mu, std) tensors for your dataset
'''
#INTERACTIVE SESSION TO COMPUTE CUSTOM MEAN AND STD OF THE DATASET, UNCOMMENT WHEN NEEDED
all_data = ConcatDataset([train_dataset, test_dataset])

all_data_loader = DataLoader(all_data, batch_size=50, shuffle=True, num_workers=2)
print(len(all_data_loader))
#here the tensor in all_data_loader has shape(757, 3, 224, 224)
for giga_tensor in all_data_loader:
    mean_giga_tensor = giga_tensor.mean(dim=(0, 2, 3))
    std_giga_tensor = giga_tensor.std(dim=(0, 2, 3))
    #here we sum over the 0 dimension therefore the huge tensor has shape(3, 224, 224) because we sum over batch
print(mean_giga_tensor) #tensor([-0.6937, -0.5052, -0.5165])
print(type(mean_giga_tensor))
print(mean_giga_tensor.shape)
print(std_giga_tensor) #tensor([1.3941, 1.2698, 1.3243])
print(type(std_giga_tensor))
print(std_giga_tensor.shape)
'''