import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.optim as optim
from torch.optim import Adam
import torchvision.models as models
import os

vgg19_pretrained = models.vgg19_bn()
print(vgg19_pretrained)
x = torch.randn(1, 3, 224, 224)

for i, layer in enumerate(vgg19_pretrained.features):
    x = layer(x)
    if isinstance(layer, nn.BatchNorm2d):
        print(
            f"layer {i}: num_features={layer.num_features}, "
            f"output shape={tuple(x.shape)}"
        )

OMP_NUM_THREADS=1

#VGG19
class VGG19_representations(nn.Module):
     model_path = "/leonardo/home/userexternal/ldepaoli/models/vgg19_bn-c79401a0.pth"
     def __init__(self): #images
          super().__init__() #refers to the class that this class inherits from (nn.Modules)
          self.vgg19_pretrained = models.vgg19_bn()
          state_dict = torch.load(VGG19_representations.model_path)
          self.vgg19_pretrained.load_state_dict(state_dict)
          self.vgg19_pretrained.to("cuda")
          self.vgg19_pretrained.requires_grad_(False) #to not compute gradients with respect to model parameters
          self.hooks = []
          self.feature_maps = {}

          count = 0
          #for (name, layer) in vgg19.features.named_modules(): #this is useful to extract only the ReLU of the of the conv not of the classifier
          for (idx, layer) in enumerate(self.vgg19_pretrained.features):
               #print(idx)
               #print(layer)
               if isinstance(layer, torch.nn.BatchNorm2d):
                    if idx in [1, 8, 15, 28, 41]:
                    #if prev_layer is None or isinstance(prev_layer, nn.MaxPool2d):
                         hook = layer.register_forward_hook(self.hook_func)
                         layer.name = f'layer_{count}'
                         self.hooks.append(hook)
               #prev_layer = layer
               count += 1

     def hook_func(self, module, input, output): #all of the three arguments are necessary
          name = module.name
          #print(f"Hook fired for: {name}")
          #feature_maps = self.feature_maps
          self.feature_maps[name] = output#.detach()

          
     def gram_matrix(self, feature_map):
          #for tensor in feature_map:
          gram_matrix = torch.einsum("bihw,bjhw->bij", feature_map, feature_map)
          
          return gram_matrix
     
     def forward(self, images):
          gram_matrix_list = []
          feature_map_m_list = []
          feature_map_n_list = []
          feature_map_list = []
          model = self.vgg19_pretrained(images)

          #check if gram matrices (list) are not transferred from gpu to cpu from time to time
          #this could slow down the whole learning process
          for key in self.feature_maps:
               feature_map = self.feature_maps[key]
               feature_map_height = feature_map.size(2)
               feature_map_width = feature_map.size(3)
               feature_map_m = feature_map_height*feature_map_width
               feature_map_m_list.append(feature_map_m)
               gram_matrices = self.gram_matrix(feature_map)
               gram_matrix_list.append(gram_matrices)
               feature_map_list.append(feature_map)
               
          return gram_matrix_list, feature_map_m_list, feature_map_list
     
def gaussian_image_tensor(size=400, mean=0.5, std=0.2):
    gaussian_image = torch.randn(3, size, size) * std + mean
    gaussian_image.clamp_(0.0, 1.0)
    return gaussian_image.to("cuda")#.unsqueeze(0)
