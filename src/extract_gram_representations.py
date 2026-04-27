import torch
import torchvision.utils as u
import torchvision.transforms as transforms
from dataloader_dtd import class_loaders
from dataloader_gaussian import gaussian_loader
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import gc
import h5py
from pathlib import Path
import argparse

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

mode_choice = "orig"

#parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=model_dict.keys(), 
                    help="Which model to run")
parser.add_argument("--mode", type=str, required=True, choices=mode_choice, 
                    help="reco or orig mode selection")

args = parser.parse_args()
model_name = args.model
model = model_dict[args.model]()

MSE = torch.nn.MSELoss()
device = "cuda"
optim_steps = 30000 #1 if in orig mode; 30000 if in reco mode
mode = mode_choice
subset = ""

#output paths
orig_path = f"/your/path/orig_images_{model_name}"
info_plot_path = f"/your/path/{mode}/info_plot_{model_name}"
gram_matrices_path = f"/your/path/{mode}_gram_{model_name}_data{subset}.h5"
checkpoint_path = f"/your/path/models_checkpoints/{mode}/class_ckpts_{model_name}"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)


for d in [orig_path, info_plot_path, gram_matrices_path, checkpoint_path]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(gram_matrices_path):
    with h5py.File(gram_matrices_path, "w") as _:
        pass

#dummy forward
_ = model(torch.randn(1,3,224,224, device=device))
layer_names = list(model.feature_maps.keys())

model.zero_grad()
model.eval()
for param in model.parameters():
    param.requires_grad_(False)

#parameters for images
mean = (0.5137, 0.4639, 0.4261)
std = (0.2576, 0.2330, 0.2412)

#parameters for dataloader
batch_size = 8

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

data_gram_list = []

h5f = h5py.File(gram_matrices_path, "a")

for texture_name, texture_loader in class_loaders.items():
    class_checkpoint_path = os.path.join(checkpoint_path, f"{texture_name}.pt")

    checkpoint = None
    start_batch = 0
    #load checkpoint if present
    if os.path.exists(class_checkpoint_path):
        checkpoint = torch.load(class_checkpoint_path)
        if checkpoint is not None and "batch_idx" in checkpoint:
            start_batch = int(checkpoint["batch_idx"])
    else: 
        print(f"Checkpoint not found: {checkpoint_path}")        

    last_batch = None
    last_loss = None
    final_reco_image = None

    for batch, (orig_image, reco_image) in enumerate(zip(texture_loader, gaussian_loader)):
        if batch < start_batch:
            continue  #skip already-completed batches for this class

        orig_image = orig_image.to(device)
        reco_image = nn.Parameter(reco_image.clone().detach().to(device)) #define gradient with respect to image as a parameter
        optimizer = optim.Adam([reco_image], lr=1e-4)

        start_step = 0
        if checkpoint is not None and batch == start_batch:
            if checkpoint["reco_image"] is not None:
                with torch.no_grad():
                    reco_image.copy_(checkpoint["reco_image"].to(device))
            if checkpoint["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["step"]
            if start_step > 0:
                print(f"Resumed checkpoint: class={texture_name} batch={batch} step={start_step}")

        with torch.no_grad():
            orig_gram_matrices, feature_map_m_list, orig_feature_map_list = model(orig_image)

        last_step = -1

        #optimization
        for step in range(start_step, optim_steps):
            last_step = step
            optimizer.zero_grad()
            reco_gram_matrices, _, reco_feature_map_list = model(reco_image)

            sum_gram_matrix_loss = 0
            for orig_gram_matrix, reco_gram_matrix, m in zip(orig_gram_matrices, reco_gram_matrices, feature_map_m_list):
                gram_matrix_loss = MSE(orig_gram_matrix, reco_gram_matrix)/(4*m)
                sum_gram_matrix_loss += gram_matrix_loss

            #backward pass over the images and update optimizer
            sum_gram_matrix_loss.backward()
            optimizer.step()

            #create dataframes for the plots
            data_gram = {
                "optim_step": step,
                "texture": texture_name,
                "loss": sum_gram_matrix_loss.item()
            }

            csv_path = os.path.join(info_plot_path, f"gram_losses_{texture_name}_s{optim_steps}.csv")
            df_gram = pd.DataFrame([data_gram])
            if os.path.exists(csv_path):
                df_gram.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df_gram.to_csv(csv_path, mode="w", header=True, index=False)      

            if step % 1000 == 0:
                torch.save({
                    "batch_idx": batch,
                    "step": step+1,
                    "reco_image": reco_image.detach().cpu(),
                    "optimizer_state_dict": optimizer.state_dict()}, class_checkpoint_path)
                print(f"Checkpoint: {texture_name} batch={batch} step={step+1} loss={sum_gram_matrix_loss.item()}", flush=True)

        step_for_name = last_step if last_step >= 0 else start_step

        with torch.no_grad():
            #denormalize orig and reco images
            denorm_image = denormalize(orig_image, mean, std).to(device)
            denorm_reco = denormalize(reco_image, mean, std).to(device)
        orig = u.save_image(denorm_image, os.path.join(orig_path, f"orig_{texture_name}_b{batch}_s{step}.png"))

        with torch.no_grad():
            reco_gram_matrices, _, _ = model(reco_image)

        for gram_idx, (orig_gram, reco_gram) in enumerate(zip(orig_gram_matrices, reco_gram_matrices)):
            B, C, _ = orig_gram.shape
            name = layer_names[gram_idx]
            for b in range(B):
                if mode == "reco":
                    g = reco_gram[b].detach().cpu().numpy()
                else:
                    continue

                base_path = f"{texture_name}/batch_{batch}/img_{b}/layer_{name}"
                group = h5f.require_group(base_path)

                if "gram" in group: #overwrite if h5py exists
                    del group["gram"]
                group.create_dataset("gram", data=g, compression="gzip", compression_opts=6)
                
        torch.save({
                "batch_idx": batch+1,
                "step": 0,
                "reco_image": reco_image.detach().cpu(),
                "optimizer_state_dict": optimizer.state_dict()}, class_checkpoint_path)
        
        last_batch = batch
        last_loss = sum_gram_matrix_loss.item() if last_step >= 0 else last_loss
        final_reco_image = reco_image

    print(f"optimization_steps: {optim_steps}, texture: {texture_name}, gram loss: {last_loss}")

    h5f.flush() 
    torch.cuda.empty_cache()
    gc.collect()

h5f.close()