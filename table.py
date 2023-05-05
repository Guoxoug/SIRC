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
from models.model_utils import MODEL_NAME_MAPPING
import os

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
    "--latex",
    type=int,
    default=1,
    help=(
        "whether to print datframe directly or to latex"
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

args = parser.parse_args()

# decide how to bold metrics
if args.problem == "SCOD":
    metrics = ["errROC", "errFPR@95"]
else:
    metrics = [" ROC", " FPR@95"] # space needed

EVAL_MAPPING = {
    "errROC": r"\%AUROC$\uparrow$",
    "ROC": r"\%AUROC$\uparrow$",
    "errFPR@95": r"\%FPR@95$\downarrow$",
    "FPR@95": r"\%FPR@95$\downarrow$"
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
    "SIRC_MSP_z",
    "SIRC_MSP_res",
    "SIRC_MSP_knn",
    "SIRC_doctor_z",
    "SIRC_doctor_res",
    "SIRC_doctor_knn",
    "SIRC_H_z",
    "SIRC_H_res",
    "SIRC_H_knn",
    "SIRC_MSP_knn_res_z",
    "SIRC_doctor_knn_res_z",
    "SIRC_H_knn_res_z",
    
    "confidence",
    "doctor",
    "entropy",
    "feature_norm",
    "residual",
    "knn",
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
    if args.latex:
        df.columns = [
            col
            if col != config["id_dataset"]["name"] else "ID\\xmark"
            for col in df.columns
        ]
    else:
        df.columns = [
            col
            if col != config["id_dataset"]["name"] else "ID - incorrect"
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
# get mean and standard deviation over runs
df = df.astype(float)
mean = df.groupby(df.index).mean()
std = df.groupby(df.index).std()
id_mean = mean[["top1", "top5", "nll"]]
id_std = std[["top1", "top5", "nll"]]


mean_std = pd.concat([mean, std])


# format
def mean_std_format(data):
    """Take array [mean, std] and return formatted string."""
    data = np.array(data)
    if args.std and args.latex:
        return f"{data[0]:.2f} \scriptsize ±{2*data[1]:.1f}"
    elif args.std:
        return f"{data[0]:.2f} ± {2*data[1]:.2f}"
    else:
        return f"{data[0]:.2f}"


dfs = []
for i in range(2):

    mean = df.groupby(df.index).mean()
    res = pd.DataFrame(mean_std.apply(mean_std_format, axis=0)).transpose()
    print("="*80)

    cols_to_drop = [
        col for col in res.columns 
        if metrics[i] not in col
    ] + ["seed"] 

    data_to_drop = [
        # optionally exclude datasets
    ]

    res = rearrange_df(res, cols_to_drop, datasets_to_drop=data_to_drop)
    mean = rearrange_df(mean, cols_to_drop, datasets_to_drop=data_to_drop)
    # exclude MD column from mean
    if args.problem == "SCOD":
        print(mean.columns)
        if args.latex:
            mean = mean.loc[:, mean.columns != "ID\\xmark"].mean(axis=1)
        else:
            mean = mean.loc[:, mean.columns != "ID - incorrect"].mean(axis=1)
        mean = mean.apply(lambda x: f"{x:.2f}")
        res.insert(1,"OOD mean",mean)
    else:
        mean = mean.mean(axis=1)

        mean = mean.apply(lambda x: f"{x:.2f}")
        res.insert(0,"OOD mean",mean)
    high = higher[i]
    def bold_max(data):
        data = list(data)
        means = [float(value.split(" ", maxsplit=1)[0]) for value in data]
        means = np.array(means)
        ids = np.argsort(means)
        if high:
            idx1 = ids[-1]
            idx2 = ids[-2]
            idx3 = ids[-3]
        else:
            idx1 = ids[0]
            idx2 = ids[1]
            idx3 = ids[2]
        data[idx1] = "\\textbf{"+ data[idx1] + "}"
        data[idx2] = "\\underline{" + data[idx2] + "}"
        data[idx3] = "\\underline{" + data[idx3] + "}"
        return data 


    res = res.loc[metrics_of_interest]

    # bold best, underline 2nd and 3rd best
    if args.latex:
        res = res.apply(bold_max, axis=0)
    res = res.transpose()

    # make text nicer
    res.columns = [
        METRIC_NAME_MAPPING[colname] 
        if colname in METRIC_NAME_MAPPING else colname.replace("_", " ")
        for colname in res.columns
    ]

    dfs.append(res)
idx = res.index
comb = pd.concat(dfs, keys=[EVAL_MAPPING[metric] for metric in metrics])
comb = comb.swaplevel().reindex(idx, level=0).transpose()


def tidy_idx_cls(df):
    """Make dataframe nicer for paper."""
    df.columns = pd.MultiIndex.from_tuples(
        [("\\textbf{" + x[0] + "}", x[1]) for x in df.columns]
    )
    # comb1 = pd.concat({config["model"]["model_type"]: comb1}, names=["model"])
    df_idx = df.index.to_frame(name="\\textbf{Method}")
    side_cat = []
    for metric in df_idx.iloc[:,0]:
        if "KNN," in metric:
            side_cat.append("\\begin{sideways}\\textbf{SIRC+}\\end{sideways}")
        elif "(" in metric:
            side_cat.append("\\begin{sideways}\\textbf{SIRC}\\end{sideways}")
        else:
            side_cat.append("")
    df_idx.insert(
        0, "", side_cat
        # [
        #     "\\begin{sideways}\\textbf{SIRC}\\end{sideways}"
        #     if "(" in metric else "" for metric in df_idx.iloc[:,0]
        # ]
    )
    df.index = pd.MultiIndex.from_frame(df_idx)
    df_idx = df.index.to_frame()


    side_cat = []

    df_idx.insert(
        0, "\\textbf{Model}",
        [
            f'\\begin{{sideways}}\\shortstack[l]{{\\textbf{{{MODEL_NAME_MAPPING[config["model"]["model_type"]]}}} \\\\ ID \\%Error: {id_mean["top1"][0]:.2f}}}\\end{{sideways}}' 
            for metric in df_idx.iloc[:,0]
        ]
    )
    df.index = pd.MultiIndex.from_frame(df_idx)

    

print(
    f"tables of results for {config['model']['model_type']}"
)    
if config["id_dataset"]["name"] in ["imagenet200"] and args.latex:
    # split in 2 for presentation purposes
    data_names1 = [
        "ID\\xmark", "OOD mean", 'Near-ImageNet-200', 
        'Caltech-45', 'Openimage-O', "iNaturalist"
    ]
    data_names2 = [
        "Textures", "SpaceNet", 'Colonoscopy', 
        'Colorectal', 'Noise', 'ImageNet-O'
    ]
    comb1 = comb[data_names1]
    comb2 = comb[data_names2]
    tidy_idx_cls(comb1)
    tidy_idx_cls(comb2)
    print(comb1.style.to_latex(hrules=True, multicol_align="c"))
    print("\n", 90*"=", "\n")
    print(comb2.style.to_latex(hrules=True, multicol_align="c"))
    print("\n", 90*"=", "\n")
    print(comb1.iloc[:,:4].style.to_latex(hrules=True, multicol_align="c"))
else:
    if args.latex:
        tidy_idx_cls(comb)
        print(comb.style.to_latex(hrules=True, multicol_align="c"))
    else:
        # print entire dataframe
        with pd.option_context(
            'display.max_rows', None, 'display.max_columns', None
        ):
            print(comb)        




