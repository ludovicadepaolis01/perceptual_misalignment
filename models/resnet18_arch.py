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

OMP_NUM_THREADS=1

resnet18_pretrained = models.resnet18()
#print(resnet18_pretrained)
def print_bn_shapes(model, input_shape=(1, 3, 224, 224)):
    hooks = []

    def hook_fn(name):
        def hook(module, inp, out):
            print(
                f"{name}: num_features={module.num_features}, "
                f"output shape={tuple(out.shape)}"
            )
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    x = torch.randn(*input_shape)
    model.eval()
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

print_bn_shapes(resnet18_pretrained)

#alexnet
class resnet18_representations(nn.Module):
     model_path = "/leonardo/home/userexternal/ldepaoli/models/resnet18-5c106cde.pth"
     def __init__(self): #images
          super().__init__() #refers to the class that this class inherits from (nn.Modules)
          self.resnet18_pretrained = models.resnet18()
          state_dict = torch.load(resnet18_representations.model_path)
          self.resnet18_pretrained.load_state_dict(state_dict)
          self.resnet18_pretrained.to("cuda")
          self.resnet18_pretrained.requires_grad_(False) #to not compute gradients with respect to model parameters
          self.hooks = []
          self.feature_maps = {}

          count = 0
          #for (name, layer) in alexnet.features.named_modules(): #this is useful to extract only the ReLU of the of the conv not of the classifier
          for (name, module) in self.resnet18_pretrained.named_modules():
               #print(name)
               #print(module)
               if name == "bn1" or name.endswith(".0.bn1") and "downsample" not in name:
                    module.name = f"bn1_{count}"
                    hook = module.register_forward_hook(self.hook_func)
                    #print(f"Registered hook on: {name} as {module.name}")
                    self.hooks.append(hook)
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
          model = self.resnet18_pretrained(images)

          #check if gram matrices (list) are not transferred from gpu to cpu from time to time
          #this could slow down the whole learning process
          #
          for key in self.feature_maps:
               feature_map = self.feature_maps[key]
               #print(f"type of feature_map: {type(feature_map)}")
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
