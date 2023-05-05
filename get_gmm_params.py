"""A vanilla version of Mahalanobis distance 
Does not include the perturbations or weighted averaging over 
features from different depths.
"""

import numpy as np
import torch
import os
import json 
import torch.nn.functional as F
import torchvision as tv


from models.model_utils import (
    model_generator, 
    load_weights_from_file,
)

from utils.data_utils import (
    Data,
    get_preprocessing_transforms
)

from utils.eval_utils import metric_stats, uncertainties

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

# mahalanobis only works with features
assert "features" in config["test_params"] and config["test_params"]["features"]


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
# bit of a hack to get around converting json syntax to bash
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
    config["test_params"]["ood_truncate"] = True
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


# print trnsforms
print("="*80)
print(id_data.name)
# change to test mode transforms without the data augmentation
id_data.train_set.transforms = id_data.test_set.transforms
print(id_data.train_set.transforms)
print("="*80)


# helper functions ------------------------------------------------------------
MAX = 250000

max_batches = int(np.ceil(MAX/id_data.train_loader.batch_size))
rem = MAX % id_data.train_loader.batch_size
# borrowed from test


def get_outputs_labels(model, loader, dev="cuda"):
    """Get the model outputs for a dataloader."""
    model.eval()
    # get ID data
    label_list = []
    logits_list = []
    feature_list = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            labels, inputs = labels.to(dev), inputs.to(dev)

            # can optionally return last hidden layer features
            outputs = model(
                inputs, return_features=True
            )

            # truncate to match MAX
            if i+1 == max_batches:
                labels, inputs = labels[:rem], inputs[:rem]
                outputs = (outputs[0][:rem], outputs[1][:rem])

            label_list.append(labels.to("cpu"))

            # in case I fuck up in specifying for the model
            logits_list.append(outputs[0][..., :id_data.num_classes].to("cpu"))
            feature_list.append(outputs[1].to("cpu"))

            if i+1 == max_batches:
                break

    labels = torch.cat(label_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    features = torch.cat(feature_list, dim=0)
    return labels, logits, features

def get_params(model, id_data, dev="cuda"):
    """Calculate statistics needed for Mahalanobis distance."""

    print(f"eval on: {id_data.name} training set")

    # this is being done on the training set this time
    labels, logits, features = get_outputs_labels(
        model, id_data.train_loader, dev=dev
    )
    print(features.shape)
    features_by_class = []
    feature_class_means = []
    # features_class_covs = []  # these are all diagonal
    print("calculating class means")
    for class_id in range(id_data.num_classes):
        mask = labels == class_id
        ids = mask.nonzero().squeeze()
        class_features = features[ids]
        class_logits = logits[ids]

        m = class_features.mean(dim=0)
        feature_class_means.append(m)
        # subtract for covariance matrix calculation
        norm_class_features = class_features - m


        features_by_class.append(
            norm_class_features
        )

    # features are now all normalised by their means
    norm_features = torch.cat(features_by_class, dim=0)

    print(f"features shape {norm_features.shape}")

    cov = norm_features.T @ norm_features / len(id_data.train_set)
    print(f"covariance shape {cov.shape}")
    # cannot use normal inverse as cov may be low-rank
    precision = torch.linalg.pinv(cov, hermitian=True)



    gmm_params_dict = {
        "class_means": feature_class_means,
        "precision":  torch.tensor(precision, dtype=torch.float32),
    }
    
    return gmm_params_dict




# evaluation-------------------------------------------------------------------

# load floating point densenet model and evaluate
model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)


# try and get weights 
if args.weights_path is not None:
    weights_path = args.weights_path
elif (
    "weights_path" in config["model"]
    and
    config["model"]["weights_path"] is not None
):
    # where pretrained weights are
    weights_path = os.path.join(
        config["model"]["weights_path"],
        get_filename(config, seed=config["seed"]) + ".pth"
    )
# try:
print(f"Trying to load weights from: {weights_path}\n")
load_weights_from_file(model, weights_path)
print("Loading successful")
# except:
#     print("Failed to load weights, will be randomly initialised.")

# multigpu
model = torch.nn.DataParallel(model) if (
    config["data_parallel"] and torch.cuda.device_count() > 1
) else model

model.to(dev)

# eval floating point model
gmm_params_dict = get_params(
    model, id_data
)


# save params

spec = get_filename(config, seed=None)
filename = get_filename(config, seed=config["seed"])
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


savepath = os.path.join(save_dir, f"{filename}_gmm{args.suffix}.pth")
torch.save(gmm_params_dict, savepath)
print(f"gmm params saved to {savepath}")


