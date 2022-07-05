"""Plot the selective classification performance as the amount of OOD data
varies."""
from sklearn.utils import shuffle
import warnings
import matplotlib as mpl
from tqdm import tqdm
from utils.train_utils import get_filename
from utils.data_utils import *
import torch
import torch.nn.functional as F
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.eval_utils import uncertainties, get_metric_name
from argparse import ArgumentParser
from sklearn.metrics._ranking import _binary_clf_curve


sns.set_theme()

parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)
parser.add_argument(
    "--logits_path",
    type=str,
    default=None,
    help=(
        "directory where result logit files are kept,"
        "deduced from config by default"
    )
)
parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)
parser.add_argument(
    "--std",
    type=int,
    default=1,
    help="whether to draw stds or not",
    choices=[0,1]
)
parser.add_argument(
    "--seeds",
    default=None,
    type=str,
    help="string containing random seeds, overrides default 1 to num_runs."
)

parser.add_argument(
    "--num_runs",
    type=int,
    default=1,
    help="number of independent runs to average over"
)

parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="filename suffix to make file unique if needs be"
)
args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# set gpu
# bit of a hack to get around converting json syntax to bash
# deals with a list of integer ids
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config["gpu_id"]
    ).replace("[", "").replace("]", "")


# list of seeds
seeds = [i for i in range(1, args.num_runs + 1)] if (
    args.seeds is None
) else list(args.seeds)

# single plot
if args.logits_path is not None:
    logits_path = args.logits_path
    logits_paths = [logits_path]
    if config["test_params"]["features"]:
        features_paths = [logits_path.replace("logits", "features")]

# results path generated as results_savedir/arch_dataset
else:
    results_path = os.path.join(
        config["test_params"]["results_savedir"],
        get_filename(config, seed=None)
    )
    logits_paths = [
        os.path.join(
            results_path, get_filename(config, seed=seed) + "_logits.pth"
        )  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth
        for seed in seeds
    ]
    if config["test_params"]["features"]:
        features_paths = [
            logits_path.replace("logits", "features")
            for logits_path in logits_paths
    ]

# these are actually dictionaries
# containing many difference quantization levels
# try and load data
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
filename = get_filename(config, seed=config["seed"]) +  \
    "_" f"plotdata"\
    + args.suffix + ".pth"
path = os.path.join(save_dir, filename)
try:
    plot_data = torch.load(path)
    generate_data = False
    print("loaded saved plot data")
except:
    plot_data = {
        "alpha":{},
        "beta":{}
    }
    generate_data = True
    print("no saved data, performing new calculations")
if generate_data:
    print("Loading logits")
    logits = [
        torch.load(path) for path in logits_paths
    ]

    if config["test_params"]["features"]:
        print("Loading features")
        features = [
            torch.load(path) for path in features_paths
        ]
    else:
        features = None


    # gmm parameters (means and covariance matrix) ------------------------
    # choose not to plot Mahalanobis 
    gmm_params = [None for seed in seeds]
    # vim parameters projection matrix etc ------------------------

    try:
        vim_paths = [
            os.path.join(
                results_path, get_filename(config, seed=seed) + "_vim.pth"
            )  # results_savedir/arch_dataset/arch_dataset_seed_vim.pth
            for seed in seeds
        ]
        vim_params = [torch.load(path) for path in vim_paths]
        print(f"vim params loaded")

    except:
        vim_params = [None for seed in seeds]
        print(f"could not load vim params")

    # stats of scores on training set
    try:
        stats_paths = [
            os.path.join(
                results_path, get_filename(config, seed=seed) + "_train_stats.pth"
            )  #
            for seed in seeds
        ]
        stats = [torch.load(path) for path in stats_paths]
        print(f"vim params loaded")

    except:
        stats = [None for seed in seeds]
        print(f"could not load vim params")


    # data stuff
    id_data = Data(
        **config["id_dataset"],
        test_only=False,
        transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
        fast=False
    )

    labels = torch.tensor(id_data.test_set.targets)

    data_to_drop = [
        # option to exclude certain datasets
    ]

    # just do floating point (legacy)
    logits = [logits[i]["afp, wfp"] for i in range(len(logits_paths))]
    features = [features[i]["afp, wfp"] for i in range(len(logits_paths))]

    ood_all = [
        key for key in logits[0].keys() 
        if key not in data_to_drop  + [id_data.name] 
    ]
    print(f"ood datasets {ood_all}")
    ood_closer = [
        "near-imagenet200", "caltech256","openimage-o", "inaturalist"
    ]

    ood_further = [
        "textures", "colorectal", "imagenet-noise","colonoscopy"
    ]

def get_uncs(logits, features=None, idx=0):
    # quick fix to filter out corrupted datasets
    logits = {k: v for k, v in logits.items() if "-c" not in k}

    if features is not None:
        data = {
            data_name: uncertainties(
                logits[data_name], features=features[data_name],
                gmm_params=gmm_params[idx], vim_params=vim_params[idx],
                stats=stats[idx]
            )
            for data_name in logits.keys()
        }
    else:
        data = {
            data_name: uncertainties(
                logits[data_name], features=None
            )
            for data_name in logits.keys()
        }

    # need to move uncertainties to the outside 
    uncs = list(data.values())[0].keys()
    data = {
        unc:{
            ds:data[ds][unc] for ds in data if unc in data[ds]
            } 
        for unc in uncs
    }
    return data

# we're combining all OOD data into one big glob for now
def get_valid_invalid_split(
    data_uncs,  labels, id_logits, shuffle_idx, ood_names=None
):
    """ID correct, ID incorrect and OOD data uncertainties.
    Takes the ood data and permutes it using shuffled indices.
    """

    preds =  id_logits.max(dim=-1).indices
    id_uncs = data_uncs[id_data.name]
    id_correct = id_uncs[preds == labels].squeeze(0)
    id_miscls =  id_uncs[preds != labels].squeeze(0)

    # join all ood data together and shuffle
    ood_uncs = torch.cat(
        [
            v for k, v in data_uncs.items() 
            if 
            k in ood_names
        ]
    )

    # shuffle idx calculated on all ood data
    # this might be a subset
    ood_uncs = ood_uncs[shuffle_idx[shuffle_idx < len(ood_uncs)]]

    return id_correct, id_miscls, ood_uncs

def risk_coverage_curve(y_true, y_score, sample_weight=None):
    
    sorted_idx = y_score.argsort(descending=True)
    # risk for each coverage value rather than recall
    # add one to cover situation with zero coverage, assume risk is zero
    # when nothing is selected
    coverage = torch.linspace(0, 1, len(y_score) + 1)
    # invert labels to get invalid predictions
    sample_costs = ~(y_true.to(bool)) * sample_weight
    sorted_cost = sample_costs[sorted_idx]
    summed_cost = torch.cumsum(sorted_cost, 0)
    n_selected = torch.arange(1, len(y_score) + 1)
    # zero risk when none selected
    risk = torch.cat([torch.zeros(1), summed_cost/n_selected])
    return risk, coverage

def risk_recall_curve(y_true, y_score, pos_label=None, sample_weight=None):
    # see https://github.com/scikit-learn/scikit-learn/blob/80598905e/sklearn/metrics/_ranking.py
    
    # the sample weight is directly multiplied with each tps/fps
    # we use this to weigh misclassifications vs correct classifications
    weighted_fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )
    # unweighted tps and fps
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=None
    )
    ps = tps + fps
    risk = np.divide(weighted_fps, ps, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]
    # zero risk at zero reall
    return np.hstack((0,risk)), np.hstack((0, recall)), thresholds


def get_eval_metrics(
    id_correct, id_micls, ood_unc, ratio, ood_weight=0.5,
    target_recall=0.95,
):
    """Get risk and AURC for a specific ratio OOD:ID and ood weight."""

    id_size = len(id_correct) + len(id_micls)
    ood_size = len(ood_unc)
    assert ratio <= ood_size/id_size
    # truncate ood data (already shuffled)
    # ratio is #ood:#id
    ood_unc = ood_unc[:int(ratio * id_size)]
    invalid = torch.cat([id_micls, ood_unc])

    # positive is ID correct now
    labels = torch.cat(
        [torch.ones(len(id_correct)), torch.zeros(len(invalid))]
    )

    sample_weights = torch.cat(
        [
            ood_weight*torch.ones(id_size), 
            torch.ones(len(ood_unc)) * (1-ood_weight)
        ]
    )
    # change to confidence/certainty
    # ordering is ID correct ID incorrect OOD
    scores = -torch.cat([id_correct, invalid])
    risk, recall, thresholds = risk_recall_curve(
        labels, scores, sample_weight=sample_weights
    )

    # get threshold closest to specified (e.g.95%) recall
    cut_off = np.argmin(np.abs(recall-target_recall))
    spec_risk = risk[cut_off]
    # auc is calculated as 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    aurr = np.sum(np.diff(recall) * np.array(risk)[1:])

    risk, coverage = risk_coverage_curve(
        labels, scores, sample_weight=sample_weights
    )

    # use trapezium rule here instead
    aurc = torch.trapz(risk, coverage).item()

    return spec_risk, aurc, aurr



uncs = [
    "confidence",
    "SIRC_H_z",
    "SIRC_H_res",
    "SIRC_MSP_z",
    "SIRC_MSP_res",
    "doctor",
    "entropy",
    "max_logit",
    "energy",
    "feature_norm",
    "residual",
    "gradnorm",
    "vim",
]

print(f"using uncertainties {uncs}")
if generate_data:

    # all OOD data
    ood_uncs = torch.cat([
        v for k, v in logits[0].items() 
        if k in ood_all
    ])

    ood_size = len(ood_uncs)
    # fix seed for reproducibility
    torch.manual_seed(0)
    # randomly ordered
    shuffle_idx = torch.randperm(ood_size)
    print(f"total OOD data: {len(shuffle_idx)}")

    ratios = np.linspace(0, 2, 21)
    betas = np.linspace(1e-6,1-1e-6,21) 
    # scikit learn function filters out zero values 


    plot_data["alpha"]["alphas"] = 1/(1+ratios)
    plot_data["beta"]["betas"] = betas
    # 3 different settings of OOD data
    # all ood datasets
    # uncertainty metric, seed, sc perf
    aurc_results = {unc:[] for unc in uncs}
    aurr_results = {unc: [] for unc in uncs}
    risk_results = {unc: [] for unc in uncs}

    # over different training runs
    for i in tqdm(range(len(logits))):
        data = get_uncs(logits[i], features[i], idx=i)
        for unc in uncs:
            id_correct, id_micls, ood = get_valid_invalid_split(
                data[unc], labels, logits[i][id_data.name], 
                shuffle_idx=shuffle_idx,
                ood_names=ood_all
            )
            if unc == "confidence":
                id_correct, id_micls, ood = -id_correct, -id_micls, -ood
            risk_list, aurc_list, aurr_list = [], [], []

            # over different ratios of ood:id
            for ratio in ratios:
                risk, aurc, aurr = get_eval_metrics(
                    id_correct, id_micls, ood, ratio, 
                )
                risk_list.append(risk)
                aurc_list.append(aurc)
                aurr_list.append(aurr)


            aurc_results[unc].append(aurc_list)
            aurr_results[unc].append(aurr_list)
            risk_results[unc].append(risk_list)

    plot_data["alpha"]["all"] = {}
    plot_data["alpha"]["all"]["aurc"] = aurc_results
    plot_data["alpha"]["all"]["aurr"] = aurr_results
    plot_data["alpha"]["all"]["risk"] = risk_results

    # beta
    aurc_results = {unc: [] for unc in uncs}
    aurr_results = {unc: [] for unc in uncs}
    risk_results = {unc: [] for unc in uncs}
    for i in tqdm(range(len(logits))):
        data = get_uncs(logits[i], features[i], idx=i)
        for unc in uncs:
            id_correct, id_micls, ood = get_valid_invalid_split(
                data[unc], labels, logits[i][id_data.name], shuffle_idx=shuffle_idx,
                ood_names=ood_all,
            )
            if unc == "confidence":
                id_correct, id_micls, ood = -id_correct, -id_micls, -ood
            risk_list, aurc_list, aurr_list = [], [], []

            # over different beta
            for beta in betas:
                risk, aurc, aurr = get_eval_metrics(
                    id_correct, id_micls, ood, ratio=1.0,
                    
                    ood_weight=beta
                )
                risk_list.append(risk)
                aurc_list.append(aurc)
                aurr_list.append(aurr)

            aurc_results[unc].append(aurc_list)
            aurr_results[unc].append(aurr_list)
            risk_results[unc].append(risk_list)

    plot_data["beta"]["all"] = {}
    plot_data["beta"]["all"]["aurc"] = aurc_results
    plot_data["beta"]["all"]["aurr"] = aurr_results
    plot_data["beta"]["all"]["risk"] = risk_results

    # 3 different settings of OOD data
    # closer ood datasets
    # uncertainty metric, seed, sc perf
    aurc_results = {unc: [] for unc in uncs}
    aurr_results = {unc: [] for unc in uncs}
    risk_results = {unc: [] for unc in uncs}
    for i in tqdm(range(len(logits))):
        data = get_uncs(logits[i], features[i], idx=i)
        for unc in uncs:
            id_correct, id_micls, ood = get_valid_invalid_split(
                data[unc], labels, logits[i][id_data.name], shuffle_idx=shuffle_idx,
                ood_names=ood_closer
            )
            if unc == "confidence":
                id_correct, id_micls, ood = -id_correct, -id_micls, -ood
            risk_list, aurc_list, aurr_list = [], [], []
            for ratio in ratios:
                risk, aurc, aurr = get_eval_metrics(
                    id_correct, id_micls, ood, ratio,
                )
                risk_list.append(risk)
                aurc_list.append(aurc)
                aurr_list.append(aurr)

            aurc_results[unc].append(aurc_list)
            aurr_results[unc].append(aurr_list)
            risk_results[unc].append(risk_list)

    plot_data["alpha"]["closer"] = {}
    plot_data["alpha"]["closer"]["aurc"] = aurc_results
    plot_data["alpha"]["closer"]["aurr"] = aurr_results
    plot_data["alpha"]["closer"]["risk"] = risk_results

    aurc_results = {unc: [] for unc in uncs}
    aurr_results = {unc: [] for unc in uncs}
    risk_results = {unc: [] for unc in uncs}
    for i in tqdm(range(len(logits))):
        data = get_uncs(logits[i], features[i], idx=i)
        for unc in uncs:
            id_correct, id_micls, ood = get_valid_invalid_split(
                data[unc], labels, logits[i][id_data.name], shuffle_idx=shuffle_idx,
                ood_names=ood_closer,
            )
            if unc == "confidence":
                id_correct, id_micls, ood = -id_correct, -id_micls, -ood
            risk_list, aurc_list, aurr_list = [], [], []
            for beta in betas:
                risk, aurc, aurr = get_eval_metrics(
                    id_correct, id_micls, ood, ratio=1.0,   
                    ood_weight=beta
                )
                risk_list.append(risk)
                aurc_list.append(aurc)
                aurr_list.append(aurr)

            aurc_results[unc].append(aurc_list)
            aurr_results[unc].append(aurr_list)
            risk_results[unc].append(risk_list)

    plot_data["beta"]["closer"] = {}
    plot_data["beta"]["closer"]["aurc"] = aurc_results
    plot_data["beta"]["closer"]["aurr"] = aurr_results
    plot_data["beta"]["closer"]["risk"] = risk_results


    # 3 different settings of OOD data
    # further ood datasets
    # uncertainty metric, seed, sc perf
    aurc_results = {unc: [] for unc in uncs}
    aurr_results = {unc: [] for unc in uncs}
    risk_results = {unc: [] for unc in uncs}
    for i in tqdm(range(len(logits))):
        data = get_uncs(logits[i], features[i], idx=i)
        for unc in uncs:
            id_correct, id_micls, ood = get_valid_invalid_split(
                data[unc], labels, logits[i][id_data.name], shuffle_idx=shuffle_idx,
                ood_names=ood_further,
            )
            if unc == "confidence":
                id_correct, id_micls, ood = -id_correct, -id_micls, -ood
            risk_list, aurc_list, aurr_list = [], [], []
            for ratio in ratios:
                risk, aurc, aurr = get_eval_metrics(
                    id_correct, id_micls, ood, ratio,
                )
                risk_list.append(risk)
                aurc_list.append(aurc)
                aurr_list.append(aurr)

            aurc_results[unc].append(aurc_list)
            aurr_results[unc].append(aurr_list)
            risk_results[unc].append(risk_list)

    plot_data["alpha"]["further"] = {}
    plot_data["alpha"]["further"]["aurc"] = aurc_results
    plot_data["alpha"]["further"]["aurr"] = aurr_results
    plot_data["alpha"]["further"]["risk"] = risk_results

    aurc_results = {unc: [] for unc in uncs}
    aurr_results = {unc: [] for unc in uncs}
    risk_results = {unc: [] for unc in uncs}
    for i in tqdm(range(len(logits))):
        data = get_uncs(logits[i], features[i], idx=i)
        for unc in uncs:
            id_correct, id_micls, ood = get_valid_invalid_split(
                data[unc], labels, logits[i][id_data.name], shuffle_idx=shuffle_idx,
                ood_names=ood_further
            )
            if unc == "confidence":
                id_correct, id_micls, ood = -id_correct, -id_micls, -ood
            risk_list, aurc_list, aurr_list = [], [], []
            for beta in betas:
                risk, aurc, aurr = get_eval_metrics(
                    id_correct, id_micls, ood, ratio=1.0,
                    
                    ood_weight=beta
                )
                risk_list.append(risk)
                aurc_list.append(aurc)
                aurr_list.append(aurr)

            aurc_results[unc].append(aurc_list)
            aurr_results[unc].append(aurr_list)
            risk_results[unc].append(risk_list)

    plot_data["beta"]["further"] = {}
    plot_data["beta"]["further"]["aurc"] = aurc_results
    plot_data["beta"]["further"]["aurr"] = aurr_results
    plot_data["beta"]["further"]["risk"] = risk_results


    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
    filename = get_filename(config, seed=config["seed"]) +  \
        "_" f"plotdata"\
        + args.suffix + ".pth"
    path = os.path.join(save_dir, filename)
    torch.save(plot_data, path)
    print(f"plot data saved to {path}")

# plot only a subset of scores to keep things readable
uncs = [
    "confidence",
    "SIRC_H_res",
    "energy",
    "vim"
]
# actual plotting here 
sns.set_palette([
    "royalblue",
    "seagreen",
    "indianred",
    "darkorange"
])
fig, axes = plt.subplots(3,4, figsize=(10,7), sharex="col", sharey="row")

def plot_uncs_over_param(ax, sc_perf, param):
    for metric in uncs:
        if uncs is not None and metric not in uncs:
            continue
        if "SIRC" in metric:
            extra = "SIRC "
        elif metric == "confidence":
            extra = "Baseline - "
        else:
            extra = ""
        unc_perf = torch.tensor(sc_perf[metric]) * 100 # readability
        metric = get_metric_name(metric).replace("\\b", "")
        if unc_perf.shape[0] > 1:
            mean = unc_perf.mean(axis=0) 
            std = unc_perf.std(axis=0) 
            ax.plot(param, mean, label=extra + f"{metric}", alpha=.5)
            if args.std:
                ax.fill_between(param, mean-1*std, mean + 1*std, alpha=0.15)
        else:
            ax.plot(ratio, unc_perf[0], label=extra + f"{metric}")

    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', )
    ax.grid(b=True, which='minor', color='w', linewidth=0.2)

EVAL_METRIC_MAPPING = {
    "risk": "Risk@95$\\leftarrow$",
    "aurr": "AURR$\\leftarrow$",
    "aurc": "AURC$\\leftarrow$",
}
eval_metrics = ["aurr", "risk", "aurc"]
for j, eval_metric in enumerate(eval_metrics):

    axes[j][0].set_ylabel(EVAL_METRIC_MAPPING[eval_metric])

    plot_uncs_over_param(
        axes[j][0], 
        plot_data["alpha"]["all"][eval_metric],
        plot_data["alpha"]["alphas"]

    )


    plot_uncs_over_param(
        axes[j][1], 
        plot_data["beta"]["all"][eval_metric],
        plot_data["beta"]["betas"]
    )

    plot_uncs_over_param(
        axes[j][2], 
        plot_data["alpha"]["closer"][eval_metric],
        plot_data["alpha"]["alphas"]

    )
    
    plot_uncs_over_param(
        axes[j][3], 
        plot_data["alpha"]["further"][eval_metric],
        plot_data["alpha"]["alphas"]

    )
    if j == 0: # along the top
        axes[j][0].set_title("$\\beta=0.5$, All OOD")
        axes[j][1].set_title("$\\alpha=0.5$, All OOD")
        axes[j][2].set_title('$\\beta=0.5$, "Close" OOD')
        axes[j][3].set_title('$\\beta=0.5$, "Far" OOD')

    if eval_metric == "aurr":
        axes[j][0].set_ylim(ymax=10)
    if j == len(eval_metrics) - 1: # only on the bottom axis
        axes[j][0].set_xlabel("$\\alpha$")
        axes[j][1].set_xlabel("$\\beta$")
        axes[j][2].set_xlabel("$\\alpha$")
        axes[j][3].set_xlabel("$\\alpha$")
        
# axes[1][0].set_xlim(xmax=0.12) # tweak
h, l = axes[0][0].get_legend_handles_labels()
order = [2,0,1,3]
order = [0,1,2,3]
h, l = [h[i] for i in order], [l[i] for i in order]
fig.tight_layout()
fig.legend(h, l, ncol=len(uncs), bbox_to_anchor=(0.5, 1), loc="upper center")
plt.subplots_adjust(top=0.85, bottom=0.1, wspace=0.1, right=.99, left=0.07)


    
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

# suffix is there for custom filename
filename = get_filename(config, seed=config["seed"]) +  \
        "_" f"_sc_vary_params_{eval_metric}"\
        + args.suffix + ".pdf"
path = os.path.join(save_dir, filename)


fig.savefig(path)
print(f"figure saved to:\n{path}")

