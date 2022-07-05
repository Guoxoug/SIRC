"""Illustrative plot to show difference between different combination methods.
Tuned for paper presentation, so quite a specific script.
"""

from models.model_utils import MODEL_NAME_MAPPING
from utils.train_utils import get_filename
import torch
import torch.nn.functional as F
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.eval_utils import (
    uncertainties, METRIC_NAME_MAPPING, 
    metric_stats, get_sirc_params, sirc
)
from argparse import ArgumentParser
from utils.data_utils import get_preprocessing_transforms, Data

parser = ArgumentParser()
parser.add_argument(
    "data",
    choices=["imagenet", "imagenet200"]
)
parser.add_argument(
    "ood",
    type=str,
    help="name of ood dataset"
)
parser.add_argument(
    "--seed",
    default=1,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)
parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="filename suffix to make file unique if needs be"
)

args = parser.parse_args()
sns.set_theme()
if args.data == "imagenet200":
    config_paths = [
        "experiment_configs/resnet50_imagenet200.json",
        "experiment_configs/mobilenetv2_imagenet200.json",
        "experiment_configs/densenet121_imagenet200.json",
    ]
else:
    config_paths = [
        "experiment_configs/resnet101v2_imagenet.json",
        "experiment_configs/densenet121_imagenet.json",
    ]

configs = [json.load(open(config_path)) for config_path in config_paths]

S1s = ["confidence", "doctor", "entropy"]
S2s = ["feature_norm", "residual"]
Ss = [[S1,S2] for S2 in S2s for S1 in S1s]
print(f"using scores {Ss}")
fig, axes = plt.subplots(
    len(config_paths), len(Ss), figsize=(15, 2.5*len(config_paths)+0.5)
)
if len(config_paths) == 1:
    axes = [axes]
for i, config in enumerate(configs):

    # training set stats ------------------------

    try:
        results_path = os.path.join(
            config["test_params"]["results_savedir"],
            get_filename(config, seed=None)
        )
        stats_path = os.path.join(
            results_path,
            get_filename(config, seed=args.seed) + "_train_stats.pth"
        )
        train_stats = torch.load(stats_path)

    except:
        train_stats = None


    results_path = os.path.join(
        config["test_params"]["results_savedir"],
        get_filename(config, seed=None)
    )
    logits_path = os.path.join(
            results_path, get_filename(config, seed=args.seed) + "_logits.pth"
        )  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth

    features_path = os.path.join(
        results_path, get_filename(config, seed=args.seed) + "_features.pth"
    )  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth

    # these are actually dictionaries
    # containing many difference quantization levels
    print("Loading logits")
    logits = torch.load(logits_path)
    print("Loading features")
    features = torch.load(features_path)


    # load vim params
    try:
        results_path = os.path.join(
            config["test_params"]["results_savedir"],
            get_filename(config, seed=None)
        )
        vim_path = os.path.join(
            results_path,
            get_filename(config, seed=args.seed) + "_vim.pth"
        )
        vim_params = torch.load(vim_path)

    except:
        vim_params = None

    id_data = Data(
        **config["id_dataset"],
        test_only=False,
        transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
        fast=False
    )

    labels = torch.tensor(id_data.test_set.targets)

    # just do one for now
    # just do floating point
    logits = logits["afp, wfp"]
    features = features["afp, wfp"]

    # quick fix to filter out corrupted datasets
    logits = {k: v for k, v in logits.items() if "-c" not in k}



    # logits and features
    ood_data_name = args.ood
    id_logits = logits[id_data.name]
    id_features = features[id_data.name]
    preds = id_logits.max(dim=-1).indices
    correct_logits = id_logits[preds==labels]
    incorrect_logits = id_logits[preds!=labels]
    correct_features = id_features[preds == labels]
    incorrect_features = id_features[preds != labels]
    ood_logits = logits[ood_data_name]
    ood_features = features[ood_data_name]


    # uncertainties
    correct_uncs = uncertainties(
        correct_logits, features=correct_features, vim_params=vim_params
    )
    incorrect_uncs = uncertainties(
        incorrect_logits, features=incorrect_features, vim_params=vim_params
    )
    ood_uncs = uncertainties(
        ood_logits, features=ood_features, vim_params=vim_params
    )

    id_uncs = uncertainties(
        id_logits, features=id_features, vim_params=vim_params
    )


    for j, S in enumerate(Ss):

        stats = metric_stats(correct_uncs) if train_stats is None else train_stats
        a, b = get_sirc_params(stats[S[1]])
        S1_string = METRIC_NAME_MAPPING[S[0]].replace("\\b", "")
        S2_string = METRIC_NAME_MAPPING[S[1]].replace("\\b", "")
        scale = 1 if S[0] == "confidence" else -1
        s1_max = 1 if S[0] != "entropy" else 0
        key1 = f"$S_1$ ({S1_string})"
        key2 = f"$S_2$ ({S2_string})"
        df = pd.DataFrame()
        # construct DataFrame for Seaborn
        dataset_df = pd.DataFrame(
            {
                key1: scale * correct_uncs[S[0]],
                key2: -correct_uncs[S[1]]
            }
        )
        dataset_df["data"] = "ID✓"
        df = pd.concat([df, dataset_df], ignore_index=True)
        dataset_df = pd.DataFrame(
            {
                key1: scale * incorrect_uncs[S[0]],
                key2: -incorrect_uncs[S[1]]
            }
        )
        dataset_df["data"] = "ID✗"
        df = pd.concat([df, dataset_df], ignore_index=True)
        dataset_df = pd.DataFrame(
            {
                key1: scale * ood_uncs[S[0]],
                key2: -ood_uncs[S[1]]
            }
        )
        dataset_df["data"] = "OOD"
        df = pd.concat([df, dataset_df], ignore_index=True)
        
        def contours(x, y): return sirc(torch.tensor(x), torch.tensor(y), a, b, s1_max=s1_max)
        leg = True if i==0 and j==0 else False
        sns.kdeplot(
            ax=axes[i][j],
            data=df, 
            x=key1,
            y=key2,
            hue="data",
            alpha=0.4,
            levels=5,
            common_norm=False,
            palette=["royalblue", "indianred","yellowgreen"],
            legend=leg
        )

        x = np.linspace(axes[i][j].get_xlim()[0], axes[i][j].get_xlim()[1], 1000) 
        y = np.linspace(axes[i][j].get_ylim()[0], axes[i][j].get_ylim()[1], 1000)
        print(axes[i][j].get_xlim(), axes[i][j].get_ylim())
        X, Y = np.meshgrid(x, y)
        Z = contours(X, Y)
        # spatially subsample, so contours look nicer but are not even in Z
        x_l = x[199:700:100]
        y_l = y[199:700:100]
        levels = contours(x_l, y_l)
        print(levels)
        axes[i][j].contour(
            X, Y, Z, 10, alpha=0.9, levels=levels, extend="neither"
        )
        if j == 0:
            axes[i][j].annotate(
                MODEL_NAME_MAPPING[config["model"]["model_type"]],
                (-.7, .5),
                xycoords="axes fraction",
                rotation=90,
                va="center", ha="center", color="gray"
            )

fig.tight_layout()
sns.move_legend(
    axes[0][0], loc="lower left", bbox_to_anchor=(0, 1), ncol=3, title=None
)

plt.subplots_adjust(top=len(configs)*2.5/(len(configs)*2.5+0.5))
# plt.subplots_adjust(bottom=0.2)
# suffix is there for custom filename
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"])
filename =  f"{id_data.name}_over_data_full" + f"_2D_{args.ood}"\
+ args.suffix + ".pdf"
path = os.path.join(save_dir, filename)
fig.savefig(path)
print(f"figure saved to:\n{path}")
