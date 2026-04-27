import h5py
import os
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import ndd
import pandas as pd
import argparse
from pathlib import Path
from sklearn import metrics


mode = "orig"

model_list = [
    "vgg16",
    "vgg19",
    "alexnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inceptionv3",
    "densenet121",
    "densenet169",
    "densenet201"
]

pretty_model_names = {
    "vgg16":        "VGG-16",
    "vgg19":        "VGG-19",
    "alexnet":      "AlexNet",
    "resnet18":     "ResNet18",
    "resnet34":     "ResNet34",
    "resnet50":     "ResNet50",
    "resnet101":    "ResNet101",
    "resnet152":    "ResNet152",
    "inceptionv3":  "InceptionV3",
    "densenet121":  "DenseNet-121",
    "densenet169":  "DenseNet-169",
    "densenet201":  "DenseNet-201",
}

#parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", 
                    type=str, 
                    required=True, 
                    choices=model_list, 
                    help="Which model to run")

args = parser.parse_args()
model_name = args.model
pretty_model = pretty_model_names.get(model_name, model_name)

#input path
path = "/your/path/gram_matrices"
file  = os.path.join(path, f"orig_gram_{model_name}_data.h5")

#output paths
rdms_path = "/your/path/rdms"
plot_path = "/your/path/plots"
csv_path = "/your/path/csvs_k47"
checkpoint_dir = f"/your/path/analyses_ckpts_{model_name}_k47"
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

for d in [rdms_path, plot_path, csv_path, ]:
    os.makedirs(d, exist_ok=True)

def plot_dendrogram(model, *, truncate_mode=None, p=30, ax=None, **dendro_kwargs):
    if ax is None:
        ax = plt.gca()

    children = model.children_
    n_samples = model.n_leaves_

    counts = np.zeros(children.shape[0])
    for i, (left, right) in enumerate(children):
        c = 0
        c += 1 if left  < n_samples else counts[left  - n_samples]
        c += 1 if right < n_samples else counts[right - n_samples]
        counts[i] = c

    if not hasattr(model, "distances_"):
        raise ValueError("Model has no distances_. Fit with compute_distances=True.")

    linkage_matrix = np.column_stack([children, model.distances_, counts]).astype(float)

    dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode,
        p=p,
        ax=ax,
        **dendro_kwargs
    )
    return linkage_matrix

#function that runs hierarchical clustering
#it plots: MI by found cluster
def hieararchical_clustering_by_mi(
        gram_vectors_data, 
        true_labels, 
        mi_function, 
        layer_name, 
        mode,
        real_classes=47, 
        plot=True, #set to True if you want to generate plots
        plot_path=".",
        checkpoint_dir=None
        ):

    if checkpoint_dir is None:
        raise ValueError("no checkpoint directory found")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{layer_name}_mi_ckpt.npz")

    mi_dict = {}
    label_dict = {}

    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path, allow_pickle=True)
        mi_dict.update(data["mi_dict"].item())
        labels_by_k = data["labels_by_k"].item()
        label_dict.update({int(k): labels_by_k[str(k)] for k in labels_by_k})
        done_ks = set(map(int, mi_dict.keys()))
        print(f"resume {layer_name}: found done ks: {sorted(done_ks)}")
    else:
        done_ks = set()

    all_ks = list(range(1, real_classes+1)) #2-47
    ks = [k for k in all_ks if k not in done_ks]

    for k in ks:
        model = AgglomerativeClustering(n_clusters=k, linkage='ward',compute_distances=True)
        found_clusters = model.fit_predict(gram_vectors_data)
        data_for_mi = np.column_stack([true_labels, found_clusters])
        mutual_info = mi_function(data_for_mi) / np.log(2)   
        mi_dict[k] = mutual_info
        label_dict[k] = found_clusters

        ks_done = sorted(set(done_ks) | set(mi_dict.keys())) 
        labels_by_k = {str(kk): label_dict[kk] for kk in label_dict}
        np.savez_compressed(
            ckpt_path,
            ks_done=np.array(ks_done),
            mi_dict=np.array(mi_dict),
            labels_by_k=np.array(labels_by_k)
        )
        print(f"ckpt layer {layer_name}: saved k={k}")

    if not mi_dict:
        raise ValueError("no clustering found :(")

    ks = sorted(mi_dict.keys())
    mi_values = [mi_dict[k] for k in ks]
 
    best_k = 47
    best_labels = label_dict[best_k]

    #plot MI by found cluster
    if plot:
        fname = f"{pretty_model}_{layer_name}_{mode}_k{best_k}_mi.png"
        plt.figure(figsize=(10, 6))
        plt.plot(ks, mi_values, marker='o', linestyle='-')
        plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
        plt.xlabel("N. of found clusters")
        plt.ylabel("MI")
        plt.title(f"{pretty_model} layer {layer_name} entropy per found clusters")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(rdms_path, fname))
        plt.close()

    #fit model with best k and plot
    final_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    found_clusters_ = final_model.fit_predict(gram_vectors_data)
        
    codes = true_labels
    return codes, best_k, best_labels, mi_dict, found_clusters_

#set up empty auto-expanding dictionaries for collecting data grouped by layer
vecs_by_layer = defaultdict(list)
labels_by_layer = defaultdict(list)

#open file explore and sort the data: 
#need to get one similarity matrix of 5640*5640 (gram computed on images) by mode by layer
with h5py.File(file, "r") as f:
    for texture_name in f.keys():
        texture = f[texture_name]
        for batch_name in texture.keys():
            batch = texture[batch_name]
            for image_name in batch.keys():
                image = batch[image_name]
                for layer_name in image.keys():
                    gram = image[layer_name]["gram"][()]
                    vec = gram.ravel()
                    vecs_by_layer[layer_name].append(vec)
                    labels_by_layer[layer_name].append(texture_name)

#compute similarity matrix, plot and hierarchical clustering
results = []  
for layer_vectors, layer_labels in [(vecs_by_layer, labels_by_layer)]:
    for layer, vectors in layer_vectors.items():
        labels = layer_labels[layer]
        X = np.stack(vectors, axis=0)
        distance_matrix = metrics.pairwise_distances(X, metric='euclidean')

        #save similarity matrix
        matrix_path = os.path.join(rdms_path, f"{model_name}_{layer}_{mode}_cosine.npy")
        np.save(matrix_path, distance_matrix)

        #plot similarity matrix heatmap 
        plt.figure(figsize=(15, 15))
        im = plt.imshow(distance_matrix, aspect="auto", vmin=0)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=40)
        plt.savefig(os.path.join(rdms_path, f"{model_name}_{layer}_{mode}_rsa.png"))
        plt.tight_layout()
        plt.close()

        codes, _ = pd.factorize(np.asanyarray(labels))

        codes, best_k, best_labels, mi_dict, found_clusters_ = hieararchical_clustering_by_mi(
            distance_matrix, 
            codes,
            mi_function=ndd.mutual_information, 
            layer_name=layer, 
            mode=mode,
            plot=True,
            plot_path=plot_path,
            checkpoint_dir=checkpoint_dir
        )
        
        clusters = np.asarray(found_clusters_).reshape(-1)

        csv_clusters = os.path.join(csv_path, f"{model_name}_{layer}_found_clusters_k47.csv")
        csv_classses = os.path.join(csv_path, f"{model_name}_{layer}_real_classes_k47.csv")
        pd.DataFrame({"cluster_id": clusters}).to_csv(csv_clusters, index=False)
        pd.DataFrame({"label": labels, "true_classes": codes}).to_csv(csv_classses, index=False)
        