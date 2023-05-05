
import torch
import torch.nn as nn
import torchvision as tv
import os
import json 
import pandas as pd
from models.model_utils import *
from utils.eval_utils import *
from utils.data_utils import (
    Data,
    get_preprocessing_transforms,
)
from tqdm import tqdm
from argparse import ArgumentParser
from utils.train_utils import get_filename

# argument parsing
parser = ArgumentParser()

parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)

parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)
parser.add_argument(
    "--weights_path",
    type=str,
    default=None,
    help="Optional path to weights, overrides config."
)

parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)

parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="added to end of filenames to differentiate them if needs be"
)



args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# set random seed
# prioritize arg seed
if args.seed is not None:
    torch.manual_seed(args.seed)
    # add seed into config dictionary
    config["seed"] = args.seed
elif "seed" in config and type(config["seed"]) == int:
    torch.manual_seed(config['seed'])
else:
    torch.manual_seed(0)
    config["seed"] = 0


# set gpu
# bit of a hack to get around converting json syntax 
# deals with a list of integer ids
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config["gpu_id"]
    ).replace("[", "").replace("]", "")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {dev}")

# ood data truncation
if "ood_truncate" not in config["test_params"]:
    config["test_params"]["ood_truncate"] = False
ood_truncate = config["test_params"]["ood_truncate"]


# data-------------------------------------------------------------------------

if config["model"]["model_type"] == "resnet101v2":  # special case
    trans = {
        "train": tv.transforms.Compose([
            tv.transforms.Resize((480, 480)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "test": tv.transforms.Compose([
            tv.transforms.Resize((480, 480)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }
else:
    trans = get_preprocessing_transforms(config["id_dataset"]["name"])

id_data = Data(
    **config["id_dataset"],
    test_only=False,
    transforms=trans,
    fast=False
)

test_loader = id_data.test_loader

# ood_data
# get id dataset normalisation values
if "ood_datasets" in config:
    if config["model"]["model_type"] == "resnet101v2": # special case
        ood_data = [
            Data(
                **ood_config,
                transforms=trans
            )
            for ood_config in config["ood_datasets"]
        ]
    else:
        ood_data = [
            Data(
                **ood_config,
                transforms=get_preprocessing_transforms(
                    ood_config["name"],
                    id_dataset_name=config["id_dataset"]["name"]
                )
            )
            for ood_config in config["ood_datasets"]
        ]
else:
    ood_data = None


# print transforms
print("="*80)
print(id_data.name)
print(id_data.test_set.transforms)
print("="*80)
for data in ood_data:
    print("="*80)
    print(data.name)
    try:
        print(data.test_set.dataset.transforms)
    except:
        print(data.test_set.transforms)
    print("="*80)


# gmm parameters (means and covariance matrix) ------------------------

try:
    results_path = os.path.join(
        config["test_params"]["results_savedir"],
        get_filename(config, seed=None)
    )
    gmm_path = os.path.join(
        results_path, 
        get_filename(config, seed=config["seed"]) + "_gmm.pth"
    )
    gmm_params = torch.load(gmm_path)

except:
    gmm_params = None


try:
    results_path = os.path.join(
        config["test_params"]["results_savedir"],
        get_filename(config, seed=None)
    )
    vim_path = os.path.join(
        results_path, 
        get_filename(config, seed=config["seed"]) + "_vim.pth"
    )
    vim_params = torch.load(vim_path)

except:
    vim_params = None


# knn params -----------------------------------------------------------------
try:
    results_path = os.path.join(
        config["test_params"]["results_savedir"],
        get_filename(config, seed=None)
    )
    knn_path = os.path.join(
        results_path,
        get_filename(config, seed=config["seed"]) + "_knn.pth"
    )
    knn_params = torch.load(knn_path)

except:
    knn_params = None


# training set stats ------------------------
# these are for SIRC
try:
    results_path = os.path.join(
        config["test_params"]["results_savedir"],
        get_filename(config, seed=None)
    )
    stats_path = os.path.join(
        results_path,
        get_filename(config, seed=config["seed"]) + "_train_stats.pth"
    )
    train_stats = torch.load(stats_path)

except:
    train_stats = None

# helper functions ------------------------------------------------------------

def get_logits_labels(
    model, loader, 
    dev="cuda", 
    early_stop=None # stop eval early 
):
    """Get the model outputs for a dataloader."""

    model.eval()
    # get ID data
    label_list = []
    logit_list = []
    feature_list = []
    count = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            labels, inputs = labels.to(dev), inputs.to(dev)
            batch_size = inputs.shape[0]
            # can optionally return last hidden layer features
            try:
                outputs = model(
                    inputs, 
                    return_features=config["test_params"]["features"]
                )
            except:
                outputs = model(
                    inputs,
                )
            label_list.append(labels.to("cpu"))
            if config["test_params"]["features"]:
                # in case I fuck up in specifying for the model
                logit_list.append(
                    outputs[0].to("cpu")[...,:id_data.num_classes]
                )
                feature_list.append(outputs[1].to("cpu"))
            else:
                logit_list.append(outputs.to("cpu"))

            count += batch_size
            if (
                early_stop is not None 
                and 
                count >= early_stop
            ):
                break

    logits, labels = torch.cat(logit_list, dim=0), torch.cat(label_list, dim=0)
    # clip to exactly match the early stop
    if early_stop is not None:
        logits, labels = logits[:early_stop], labels[:early_stop]
    if not config["test_params"]["features"]:
        return logits, labels
    else:
        features = torch.cat(feature_list, dim=0)
        if early_stop is not None:
            features = features[:early_stop]
            print(logits.shape)
        return logits, labels, features


def evaluate(
    model, id_data, 
    ood_data=None, dev="cuda",
    shifted=False
):
    """Evaluate the model's topk error rate and ECE."""
    top1 = TopKError(k=1, percent=True)
    top5 = TopKError(k=5, percent=True)
    nll = nn.CrossEntropyLoss()

    logits_dict = {}
    features_dict ={}
    print(f"eval on: {id_data.name}")
    # gradnorm uses the features from the final hidden layer
    if config["test_params"]["features"]:
        logits, labels, features = get_logits_labels(
            model, id_data.test_loader, dev=dev,
        )
    else:
        logits, labels = get_logits_labels(
            model, id_data.test_loader, dev=dev,
        )

        features = None


    # store logits for later
    logits_dict[f"{id_data.name}"] = logits.to("cpu")

    if features is not None:
        features_dict[f"{id_data.name}"] = features.to("cpu")

    results = {}
    results["dataset"] = id_data.name
    results["top1"] = top1(labels, logits)
    results["top5"] = top5(labels, logits)
    results["nll"] = nll(logits, labels).item() # backwards

   
    # average uncertainties
    metrics = uncertainties(
        logits, 
        features=features, 
        gmm_params=gmm_params,
        vim_params=vim_params, knn_params=knn_params,
        stats=train_stats
    )

    # record average values 
    res = {
        f"{id_data.name} {k}": v.mean().item()
        for k, v in metrics.items()
    }
    results.update(res)


    # ID correct vs incorrect
    max_logits, preds = logits.max(dim=-1)
    miscls_labels = (preds != labels)

    # AUROC
    miscls_res = detect_results(
        miscls_labels, metrics, mode="ROC"
    )
    miscls_res = {
        f"{id_data.name} errROC " + k: v for k, v in miscls_res.items() if k != "mode"
    }
    results.update(miscls_res)
    # FPR@95
    miscls_res = detect_results(
        miscls_labels, metrics, mode="FPR@95"
    )
    miscls_res = {
        f"{id_data.name} errFPR@95 " + k: v for k, v in miscls_res.items() if k != "mode"
    }
    results.update(miscls_res)
    correct_idx = (miscls_labels == 0).nonzero().squeeze(-1)


    # OOD data stuff
    if ood_data is not None and config["test_params"]["ood_data"]:
        ood_results = {}
        for data in ood_data:
            print(f"eval on: {data.name}")
            if config["test_params"]["features"]:
                ood_logits, _, ood_features = get_logits_labels(
                    model, data.test_loader, dev=dev,
                )
            else:
                ood_logits, _ = get_logits_labels(
                    model, data.test_loader, dev=dev,
                )
                ood_features = None

            # balance the #samples between OOD and ID data
            # unless OOD dataset is smaller than ID, then it will stay smaller
            # this does not happen by default
            if ood_truncate:
                ood_logits = ood_logits[:len(logits)]
           
            logits_dict[f"{data.name}"] = ood_logits
            
            combined_logits = torch.cat([logits, ood_logits])
            
            # ID 0, OOD 1 
            # OOD detection first
            # this gets flipped later on, we want ID to be positive
            domain_labels = torch.cat(
                [torch.zeros(len(logits)), torch.ones(len(ood_logits))]
            )

            # optional features
            if ood_features is not None:
                if ood_truncate:
                   ood_features = ood_features[:len(features)]
                features_dict[f"{data.name}"] = ood_features
                combined_features = torch.cat([features, ood_features])
            else: 
                combined_features = None
            # gets different uncertainty metrics for combined ID and OOD
            metrics = uncertainties(
                combined_logits, 
                features=combined_features, gmm_params=gmm_params,
                vim_params=vim_params, knn_params=knn_params,
                stats=train_stats
            )


            # average uncertainties
            res = {
                f"{data.name} {k}": v.mean().item()
                for k, v in metrics.items()
            }
            ood_results.update(res)

            # OOD detection
            res = detect_results(
                domain_labels, metrics, mode="ROC"
            )
            res = {
                f"OOD {data.name} ROC " + k: v
                for k, v in res.items()
                if k != "mode"
            }
            ood_results.update(res)
            res = detect_results(
                domain_labels, metrics, mode="FPR@95"
            )
            res = {
                f"OOD {data.name} FPR@95 " + k: v 
                for k, v in res.items() 
                if k != "mode"
            }
            ood_results.update(res)

            # now we treat only correct classifications as positive
            # OOD is negative class, and we get rid of ID incorrect samples

            correct_logits = logits[correct_idx]
            correct_features = features[correct_idx] if features is not None else None

            if ood_truncate:
                ood_logits = ood_logits[:len(correct_logits)]

            combined_logits = torch.cat([correct_logits, ood_logits])

            # ID correct 0, OOD 1
            domain_labels = torch.cat(
                [torch.zeros(len(correct_logits)), torch.ones(len(ood_logits))]
            )

            # optional features
            if ood_features is not None:
                if ood_truncate:
                   ood_features = ood_features[:len(correct_features)]
                combined_features = torch.cat([correct_features, ood_features])
            else:
                combined_features = None

            metrics = uncertainties(
                combined_logits,
                features=combined_features, gmm_params=gmm_params,
                vim_params=vim_params, knn_params=knn_params,
                stats=train_stats
            )

            res = detect_results(
                domain_labels, metrics, mode="ROC"
            )

            res = {
                f"{data.name} errROC " + k: v
                for k, v in res.items()
                if k != "mode"
            }
            ood_results.update(res)

            res = detect_results(
                domain_labels, metrics, mode="FPR@95"
            )

            res = {
                f"{data.name} errFPR@95 " + k: v
                for k, v in res.items()
                if k != "mode"
            }
            ood_results.update(res)

        results.update(ood_results)
    
    return results, logits_dict, features_dict


# evaluation-------------------------------------------------------------------

# load floating point densenet model and evaluate
model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)


# try and get weights 
# manual argument passed to script
if args.weights_path is not None:
    weights_path = args.weights_path

# automatically looks for weights according to config
elif (
    "weights_path" in config["model"]
    and
    config["model"]["weights_path"] is not None
):
    # where trained weights are
    weights_path = os.path.join(
        config["model"]["weights_path"],
        get_filename(config, seed=config["seed"]) + ".pth"
    )

print(f"Trying to load weights from: {weights_path}\n")
load_weights_from_file(model, weights_path)
print("Loading successful")

# multigpu
model = torch.nn.DataParallel(model) if (
    config["data_parallel"] and torch.cuda.device_count() > 1
) else model
model.to(dev)

# list of results dictionaries
result_rows = []

# eval floating point model
results, logits, features = evaluate(
    model, id_data, ood_data=ood_data
)

results["seed"] = config["seed"]
print("floating point" + 80*"=")
print_results(results)
result_rows.append(results)
print(f"datasets: {logits.keys()}")



# stored for later use
# precision here due to legacy reasons
precision_logit_dict = {}
precision_logit_dict["afp, wfp"] = logits

# save those features as well
if config["test_params"]["features"]:
    precision_feature_dict = {}
    precision_feature_dict["afp, wfp"] = features


# results into DataFrame
result_df = pd.DataFrame(result_rows)

# save to subfolder with dataset and architecture in name
# filename will have seed 
if config["test_params"]["results_save"]:
    spec = get_filename(config, seed=None)
    filename = get_filename(config, seed=config["seed"])
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    savepath = os.path.join(save_dir, f"{filename}{args.suffix}.csv")

    # just overwrite what's there
    result_df.to_csv(savepath, mode="w", header=True)
    print(f"results saved to {savepath}")

# save the logits from all precisions
if config["test_params"]["logits_save"]:
    spec = get_filename(config, seed=None)
    filename = get_filename(config, seed=config["seed"])
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    savepath = os.path.join(save_dir, f"{filename}_logits{args.suffix}.pth")
    torch.save(precision_logit_dict, savepath)
    print(f"logits saved to {savepath}")

    if config["test_params"]["features"]:
        spec = get_filename(config, seed=None)
        filename = get_filename(config, seed=config["seed"])
        save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        savepath = os.path.join(
            save_dir, f"{filename}_features{args.suffix}.pth"
        )
        torch.save(precision_feature_dict, savepath)
        print(f"features saved to {savepath}")

    

