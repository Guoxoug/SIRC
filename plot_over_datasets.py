import argparse
from utils.data_utils import DATA_NAME_MAPPING
import pandas as pd

pd.set_option("display.max_rows", 200,
              "display.max_columns", 10)
import numpy as np
from argparse import ArgumentParser
import json
from utils.train_utils import get_filename
from utils.eval_utils import METRIC_NAME_MAPPING
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_theme()

parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)
parser.add_argument(
    "num_runs",
    type=int,
    help="number of independent runs to average over"
)

parser.add_argument(
    "--results_path",
    type=str,
    default=None,
    help=(
        "directory where result .csv files are kept," 
        "deduced from config by default"
    )
)

parser.add_argument(
    "--std",
    type=int,
    default=1,
    help=(
        "whether to print stds or not"
    )
)

parser.add_argument(
    "--seeds",
    default=None,
    type=str,
    help="string containing random seeds, overrides default 1 to num_runs."
)

parser.add_argument(
    "--problem",
    default="SCOD",
    choices=["SCOD", "OOD"],
    type=str
)

parser.add_argument(
    "--suffix",
    default="",
    type=str
)

args = parser.parse_args()

# decide how to bold metrics
if args.problem == "SCOD":
    metrics = ["errROC", "errFPR@95"]
else:
    metrics = [" ROC", " FPR@95"] # space needed

EVAL_MAPPING = {
    "errROC": r"$\Delta$%AUROC$\rightarrow$",
    "ROC": r"$\Delta$%AUROC$\rightarrow$",
    "errFPR@95": r"$\Delta$%FPR@95$\leftarrow$",
    "FPR@95": r"$\Delta$%FPR@95$\leftarrow$"
}
higher = [True, False]

# load config
config = open(args.config_path)
config = json.load(config)

# list of seeds
seeds = [i for i in range(1, args.num_runs + 1)] if (
    args.seeds is None
) else list(args.seeds)

# results path generated as results_savedir/arch_dataset
if args.results_path is not None:
    results_path = args.results_path
else:
    results_path = os.path.join(
        config["test_params"]["results_savedir"], 
        get_filename(config, seed=None)
    )


# metrics we care about
metrics_of_interest = [
    "entropy",
    "SIRC_H_z",
    "SIRC_H_res",
    "SIRC_H_knn",
    "SIRC_H_knn_res_z",

]

# metrics_of_interest = [
#     "confidence",
#     "SIRC_MSP_z",
#     "SIRC_MSP_res",
#     "SIRC_MSP_knn",
#     "SIRC_MSP_knn_res_z",

# ]


# reformatting function
def rearrange_df(df, cols_to_drop, datasets_to_drop=[]):

    # get rid of specified colums
    df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    # get rid of certain data
    data_cols_to_drop = [
        col for col in df.columns
        if
        any(data_name in col for data_name in datasets_to_drop)
        and "fix" not in col
        or "-c" in col
        or "-r" in col
        
    ]
    df.drop(data_cols_to_drop, axis=1, inplace=True, errors="ignore")

    # drop shifted rows
    df.dropna(axis=1, inplace=True)
    
    df = df.transpose().reset_index(level=0)
    df.columns = ["data-method", "performance"]
    df[["data", "method"]] = df["data-method"].str.rsplit(
        " ", 1, expand=True
    )

    df.drop("data-method", axis=1, inplace=True, errors="ignore")
    df = df[["data", "method", "performance"]]

    def clean_data_name(name: str):
        for pattern in [
            " PR", " ROC", "OOD ", " FPR@95", " errROC", " errFPR@95", "err@95"
        ]:
            name = name.replace(pattern, "")

        return name

    # retain order of datasets

    df["data"] = df["data"].apply(clean_data_name)
    df["data"] = pd.Categorical(
        df["data"],
        categories=df["data"].unique(),
        ordered=True
    )
    df = df.pivot(
        index="method", columns="data", values="performance"
    )
    # nice names for the datasets
    df.columns = [
        col
        if col != config["id_dataset"]["name"] else "ID\\xmark"
        for col in df.columns
    ]
    df.columns = [
        DATA_NAME_MAPPING[col]
        if col in DATA_NAME_MAPPING else col
        for col in df.columns
    ]


    return df

# get only FP row, outputs as a series so need to do a bit of messing around
dfs = [
    pd.DataFrame(
        pd.read_csv(
            os.path.join(
                results_path, get_filename(config, seed=seed) + ".csv"
            ),  # results_savedir/arch_dataset/arch_dataset_seed.csv
            index_col=0
        ).iloc[0].drop(
            ["weights", "activations", "dataset", "precision"], 
            axis=0,
            errors="ignore"
        )
    ).transpose()
    for seed in
    seeds
]

# concatenate into a big boi
df = pd.concat(dfs)
df.drop(columns=df.filter(regex = 'risk').columns, errors="ignore", inplace=True)
df.drop(columns=df.filter(regex = 'cov').columns, errors="ignore", inplace=True)

# get mean and standard deviation over runs
df = df.astype(float)

mean = df.groupby(df.index).mean()
std = df.groupby(df.index).std()
id_mean = mean[["top1", "top5", "nll"]]
id_std = std[["top1", "top5", "nll"]]


mean_std = pd.concat([mean, std])



fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True)
for i, ax in enumerate(axes):

    mean = df.groupby(df.index).mean()
    # res = pd.DataFrame(mean_std.apply(mean_std_format, axis=0)).transpose()
    res = mean.copy(deep=True)
    print("="*80)

    cols_to_drop = [
        col for col in res.columns 
        if metrics[i] not in col
    ] + ["seed"] 

    data_to_drop = [
        "danbooru", "imagenet-r", "food101", "sun", "places365", "solarcells"
    ]

    res = rearrange_df(res, cols_to_drop, datasets_to_drop=data_to_drop)
    mean = rearrange_df(mean, cols_to_drop, datasets_to_drop=data_to_drop)


    if args.problem == "SCOD":
 
        mean = mean.loc[:, mean.columns != "ID\\xmark"].mean(axis=1)

    else:
        mean = mean.mean(axis=1)
    mean = mean.loc[metrics_of_interest]
   
    high = higher[i]

    res = res.loc[metrics_of_interest]
    res = res.transpose()

    # make text nicer
    res.columns = [
        METRIC_NAME_MAPPING[colname] 
        if colname in METRIC_NAME_MAPPING else colname.replace("_", " ")
        for colname in res.columns
    ]

    # print(res)

    
    x_labels = res.index.tolist()
    x_labels[0] = "ID✘"
    res.index=x_labels
    scores = [score.replace("\\b", "") for score in res.columns]
    res.columns = scores
    for j,col in enumerate(res.columns):
        if j == 0:
            col1 = col
            continue
        alpha = 1  if j == len(res.columns) -1 else 0.6

        if j == len(res.columns) -1:
            label = f"{scores[j]}"
        else:
            label = scores[j]
        if j ==1:
            empty, = ax.plot(
                [],[],
                label="SIRC",
                linestyle=""
            )
        if j == len(res.columns) -1:
            empty, = ax.plot(
                [],[],
                label="SIRC+",
                linestyle=""
            )
        if j == 1 or j == len(res.columns) - 1:
            ax.plot(
                x_labels,
                res[col] - res[col1],
                label=label,
                linestyle="-",
                marker="o", 
                alpha=alpha,
                color=empty.get_color()
            )
        else:
            ax.plot(
                x_labels,
                res[col] - res[col1],
                label=label,
                linestyle="-",
                marker="o", 
                alpha=alpha
            )
        ax.set_xticklabels(["\n"*(i%2) + l for i,l in enumerate(x_labels)])
    if i == 0:
        ax.legend()
    ax.set_ylabel(f"{EVAL_MAPPING[metrics[i]]}\n(from {scores[0]})")
    ax.minorticks_on()

    ax.grid(visible=True, which='minor', color='w', linewidth=0.5, axis="both")
    fig.tight_layout()
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
    filename = get_filename(config, seed=config["seed"]) +  \
    f"_over_data_{metrics[i]}_{metrics_of_interest[0]}" + f"_2D"\
    + args.suffix + ".pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    print(f"figure saved to:\n{path}")


# single score for illustrative purposes

fig, ax = plt.subplots(1,1, figsize=(5,4))

res=res.sub(
    res[scores[0]], axis=0
).drop([scores[0],scores[-1]], axis=1, errors="ignore")

x_labels = [x_labels[i] for i in [0,3,4,6,7,-1]]
res.index.name="data"
res = res.iloc[[0,3,4,6,7,-1]]
res.reset_index(inplace=True)
res.columns.name="method"
# print(res)
val_name = f"{EVAL_MAPPING[metrics[-1]]}\n(from {scores[0]})"

res = res.melt(
    id_vars=["data"], 
    var_name="method", 
    value_name=val_name
        
)
sns.barplot(
    ax=ax,
    data=res,
    x="data",
    y=val_name,
    hue="method",
    alpha=0.5,
)

ax.set_xticklabels(["\n"*(i%2) + l for i,l in enumerate(x_labels)])
ax.set_title(f"SIRC vs {scores[0]}")
ax.legend()
ax.set_xlabel("")
ax.minorticks_on()
ax.grid(visible=True, which='minor', color='w', linewidth=0.5, axis="both")
fig.tight_layout()
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
filename = get_filename(config, seed=config["seed"]) +  \
f"_over_data_{metrics[i]}_{metrics_of_interest[0]}" + f"_single"\
+ args.suffix + ".pdf"
path = os.path.join(save_dir, filename)
fig.savefig(path)
print(f"figure saved to:\n{path}")





