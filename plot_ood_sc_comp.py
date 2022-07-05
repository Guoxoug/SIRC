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
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

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
    "--seeds",
    default=None,
    type=str,
    help="string containing random seeds, overrides default 1 to num_runs."
)


args = parser.parse_args()

metrics = ["errFPR@95", " FPR@95"]

EVAL_MAPPING = {
    "errROC": r"\%AUROC$\uparrow$",
    "ROC": r"\%AUROC$\uparrow$",
    "errFPR@95": r"FPR@95$\downarrow$",
    "FPR@95": r"FPR@95$\downarrow$"
}

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
    "SIRC_MSP_z",
    "SIRC_MSP_res",
    "SIRC_doctor_z",
    "SIRC_doctor_res",
    "SIRC_H_z",
    "SIRC_H_res",
    "confidence",
    "doctor",
    "entropy",
    "feature_norm",
    "residual",
    "max_logit",
    "energy",
    "gradnorm",
    "vim",
    "mahalanobis",    
]


# reformatting function
def rearrange_df(df, cols_to_drop, datasets_to_drop=[]):

    # get rid of specified colums
    df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    # get rid of certain data
    data_cols_to_drop = [
        col for col in df.columns
        if
        any(data_name in col for data_name in datasets_to_drop)
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

# get only 1st row, outputs as a series so need to do a bit of messing around
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

# concatenate together
df = pd.concat(dfs)
# get mean and standard deviation over training runs
df = df.astype(float)
mean = df.groupby(df.index).mean()
std = df.groupby(df.index).std()
id_mean = mean[["top1", "top5", "nll"]]
id_std = std[["top1", "top5", "nll"]]


mean_std = pd.concat([mean, std])

dfs = []
for i in range(2):

    res = df.groupby(df.index).mean()
    mean = df.groupby(df.index).mean()
    print("="*80)

    cols_to_drop = [
        col for col in res.columns 
        if metrics[i] not in col
    ] + ["seed"] 

    data_to_drop = [
        # option to exclude certain datasets
    ]

    res = rearrange_df(res, cols_to_drop, datasets_to_drop=data_to_drop)
    mean = rearrange_df(mean, cols_to_drop, datasets_to_drop=data_to_drop)
    # exclude MD column from mean
    if i==0:
        print(mean.columns)
        mean = mean.loc[:, mean.columns != "ID\\xmark"].mean(axis=1)
        res.insert(1,"OOD mean",mean)
    else:
        mean = mean.mean(axis=1)
        res.insert(0,"OOD Detection mean",mean)
    res = res.loc[metrics_of_interest]
    res = res.transpose()
    if not i:
        res = res.loc[["ID\\xmark", "OOD mean"]]
    else:
        res = pd.DataFrame(res.loc["OOD Detection mean"]).transpose()

    # make text nicer
    res.columns = [
        METRIC_NAME_MAPPING[colname] 
        if colname in METRIC_NAME_MAPPING else colname.replace("_", " ")
        for colname in res.columns
    ]
    dfs.append(res)
idx = res.index
res = pd.concat(dfs, axis=0)

# get rid of components 
# keep only full detection methods
res = res.sub(res["MSP"], axis=0).drop(
    ["MSP", "Residual", "$||\\b z||_1$"], axis=1
)
res.index.name = "separation"
res.reset_index(inplace=True)
res.loc[len(res)] = 0.0
res.loc[len(res)] = 0.0
res.loc[len(res)] = 0.0
res.iloc[5, 0] = 1.0
res.iloc[4, 0] = 2.0
# this is just inserting a gap into the barplot
res = res.reindex([5, 0,1, 3,2,4])

res = res.melt(
    id_vars=["separation"], var_name="method", value_name="$\\Delta$%FPR@95$\leftarrow$\n(from MSP baseline)"
)
def clean_strings(x):
    mapping = {
        "ID\\xmark": "ID✗|ID✓",
        "OOD mean": "OOD|ID✓",
        "OOD Detection mean": "OOD|ID"
    }
    if x in mapping:
        return mapping[x]
    elif type(x) == str:
        return x.replace("\\b", "")
    else: 
        return x
res["separation"] = res["separation"].apply(clean_strings)
res["method"] = res["method"].apply(clean_strings)

sns.set_theme()
fig, ax = plt.subplots(1,1,figsize=(15,2.5))

sns.barplot(
    ax=ax,
    data=res,
    x="method",
    y="$\\Delta$%FPR@95$\leftarrow$\n(from MSP baseline)",
    hue="separation",
    palette=["white","indianred", "yellowgreen", "white", "darkolivegreen",],
    alpha=0.5,
)

h, l = ax.get_legend_handles_labels()
ax.legend(h[1:3] + [h[4]], l[1:3]+[l[4]])
sns.move_legend(
    ax,
    "lower center",
    bbox_to_anchor=(.5, 0.95), ncol=3, title=None, frameon=False,
)
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.grid(b=True, which='minor', lw=0.5)
ax.annotate(
    'SIRC', 
    xy=(0.23, .55), 
    xytext=(0.23, .7), 
    xycoords='axes fraction', 
    ha='center', va='bottom',
    arrowprops=dict(
                arrowstyle='-[, widthB=18, lengthB=1.0', 
                lw=2, 
                color="slategrey"
        ), 
    color="slategrey"

)


fig.tight_layout()

 # specify filename
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
filename = get_filename(config, seed=config["seed"]) +  \
    f"_bar_ood_sc_full.pdf"
path = os.path.join(save_dir, filename)
fig.savefig(path)
print(f"figure saved to:\n{path}")
