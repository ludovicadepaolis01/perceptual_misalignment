import os
import numpy as np
import ndd
import glob
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import matplotlib as mpl
from natsort import natsorted

#input paths
gram_matrices_path = "your/path/csvs_k47"

#output paths
out_path = f"/your/path/csvs"
plot_path = f"your_path/plots"

for d in [out_path, plot_path]:
    os.makedirs(d, exist_ok=True)

#params
info_metric = "bits"

all_files = glob.glob(os.path.join(gram_matrices_path, "*.csv"))

files_classes  = natsorted([f for f in all_files if "real_classes"   in os.path.basename(f)])
files_clusters = natsorted([f for f in all_files if "found_clusters" in os.path.basename(f)])

pretty_model_names = {
    "alexnet":      "AlexNet",
    "densenet121":  "DenseNet-121",
    "densenet169":  "DenseNet-169",
    "densenet201":  "DenseNet-201",
    "inceptionv3":  "InceptionV3",
    "resnet18":     "ResNet18",
    "resnet34":     "ResNet34",
    "resnet50":     "ResNet50",
    "resnet101":    "ResNet101",
    "resnet152":    "ResNet152",
    "vgg16":        "VGG-16",
    "vgg19":        "VGG-19",
}

def mi_estimate(
        classes,
        clusters,
        output_path,
        plot=True,
        plot_path=".",
        ):
    
    data = [] 
    for class_, cluster in zip(classes, clusters): 
        df_classes = pd.read_csv(class_) 
        df_clusters = pd.read_csv(cluster) 
        df_joint = pd.DataFrame({"true_classes": df_classes["true_classes"], "cluster_id": df_clusters["cluster_id"] }) 

        #convert MI in bits
        mi_nats = ndd.mutual_information(df_joint.to_numpy(dtype=int)) 
        mi_bits = mi_nats * np.log2(np.e) 

        class_string = os.path.basename(class_) 
        
        model_pattern = re.compile(r"^([^_]+)") 
        layer_pattern = re.compile(r"layer_(?:layer|bn\d*|conv)_\d+") 
        model = re.match(model_pattern, class_string).group(1) 
        layer = re.search(layer_pattern, class_string).group(0) 
        data.append({"model": model, "layer": layer, "mi": mi_bits}) 
            
    df = pd.DataFrame(data, columns=["model", "layer", "mi"]).reset_index(drop=True) 
    mi_csv = df.to_csv(os.path.join(output_path, f"mi_csv_k47.csv")) 

    df_copy = df.copy() 
    df_copy["layer_idx"] = df_copy["layer"].str.extract(r"(\d+)$").astype(int) 
    df_copy = df_copy.sort_values(["model", "layer_idx"])
    df_copy["pos"] = df_copy.groupby("model").cumcount() + 1

    if plot:
        fig, ax = plt.subplots(figsize=(11, 7),
                            constrained_layout=True)

        models = df_copy["model"].unique()
        cmap = mpl.colormaps.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, len(models)))

        for idx, model in enumerate(models):
            group = df_copy[df_copy["model"] == model]
            ax.plot(group["pos"], 
                    group["mi"], 
                    marker="o", 
                    linewidth=2.5,
                    markersize=8,
                    markeredgewidth=1.5,
                    label=pretty_model_names.get(model, model), 
                    color=colors[idx])

        ax.set_xlabel("Layer indices (1-5)", fontsize=20)
        ax.set_ylabel(f"MI ({info_metric})", fontsize=20)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.9, 5.1)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(
            title="Model", 
            loc="best", 
            fontsize=17,
            bbox_to_anchor=(1.01, 0.93), 
            )
        leg.get_title().set_fontsize(20)
        plt.savefig(os.path.join(plot_path, f"mi_per_model_data_{info_metric}_k47.png"), bbox_inches="tight")
        plt.close(fig)
    
mi_estimate(
    classes=files_classes,
    clusters=files_clusters,
    plot_path=plot_path,
    output_path=out_path
)