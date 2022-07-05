"""Run over the train set to get the required principle space and
scaling factor (related to the average max logit).

Adding average softmax values to this one as well.
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

# need features
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

if config["model"]["model_type"] == "resnet101v2": # special case
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
            logits_list.append(outputs[0][...,:id_data.num_classes].to("cpu"))
            feature_list.append(outputs[1].to("cpu"))

            if i+1 == max_batches:
                break

    labels = torch.cat(label_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    features = torch.cat(feature_list, dim=0)
    return  labels, logits, features

def get_params(model, id_data, dev="cuda"):
    """Calculate params for vim and KL matching on training set."""

    print(f"eval on: {id_data.name} training set")

    # this is being done on the training set this time
    labels, logits, features = get_outputs_labels(
        model, id_data.train_loader, dev=dev
    )

    # get final fc layer 
    if config["model"]["model_type"] == "resnet101v2": # special case
        W = model.head.conv.weight.cpu().detach().squeeze()
        b = model.head.conv.bias.cpu().detach().squeeze()
    else:
        try:
            W = model.state_dict()["fc.weight"].detach().clone().cpu()
            b = model.state_dict()["fc.bias"].detach().clone().cpu()
        except:
            W = model.state_dict()["classifier.weight"].detach().clone().cpu()
            b = model.state_dict()["classifier.bias"].detach().clone().cpu()

    u = -torch.linalg.pinv(W) @ b

    # size of subspace
    # pretty much just a heuristic
    D = 1000 if features.shape[-1] > 1500 else 512 
    if features.shape[-1] < 512:
        D = features.shape[-1]//2
    centered_feats = features - u
    U = torch.linalg.eigh(
        centered_feats.T@centered_feats
    ).eigenvectors.flip(-1)
    R = U[:,D:] # eigenvectors in columns
    assert R.shape[0] == features.shape[-1]
    vlogits = torch.norm(
        (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), 
        p=2, dim=-1
    )
    alpha = logits.max(dim=-1).values.mean() / vlogits.mean()
    alpha = alpha.item()

    vim_params_dict = {
        "alpha": alpha,
        "u":  u,
        "R": R
    }

    centered_feats = F.normalize(centered_feats, dim=-1)
    U = torch.linalg.eigh(
        centered_feats.T@centered_feats
    ).eigenvectors.flip(-1) # rev order to descending eigenvalue
    R = U[:,D:] # eigenvectors in columns
    assert R.shape[0] == features.shape[-1]
    vlogits = torch.norm(
        (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), 
        p=2, dim=-1
    )
    alpha = logits.max(dim=-1).values.mean() / vlogits.mean()
    alpha = alpha.item()
    vim_params_dict.update({
        "norm_alpha": alpha,
        "norm_R": R
    })



    # training data stats
    uncs = uncertainties(logits, features=features, vim_params=vim_params_dict)
    stats = metric_stats(uncs)
    return vim_params_dict, stats





# evaluation-------------------------------------------------------------------

# load floating point densenet model and evaluate
model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)

# early exit
# NB quantization is NOT supported
if (
    "early_exit_params" in config["model"]
    and
    config["model"]["early_exit_params"]
):
    model = add_exits(
        model, config["model"]["model_type"],
        config["model"]["model_params"],
        config["model"]["early_exit_params"]
    )

    multihead = True
    print("Using multiheaded network")


else:
    multihead = False
    print("Using single network")

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

# get final fc layer
if config["model"]["model_type"] == "resnet101v2":  # special case
    W = model.head.conv.weight.cpu().detach().squeeze()
    b = model.head.conv.bias.cpu().detach().squeeze()
else:
    try:
        W = model.state_dict()["fc.weight"].detach().clone().cpu()
        b = model.state_dict()["fc.bias"].detach().clone().cpu()
    except:
        W = model.state_dict()["classifier.weight"].detach().clone().cpu()
        b = model.state_dict()["classifier.bias"].detach().clone().cpu()
# except:
#     print("Failed to load weights, will be randomly initialised.")

# # multigpu
# model = torch.nn.DataParallel(model) if (
#     config["data_parallel"] and torch.cuda.device_count() > 1
# ) else model

model.to(dev)
# eval floating point model
vim_params_dict, training_stats = get_params(
    model, id_data
)

print(f'alpha: {vim_params_dict["alpha"]}')
print(f'u shape: {vim_params_dict["u"].shape}')
print(f'R shape: {vim_params_dict["R"].shape}')

# save params

spec = get_filename(config, seed=None)
filename = get_filename(config, seed=config["seed"])
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


savepath = os.path.join(save_dir, f"{filename}_vim{args.suffix}.pth")
torch.save(vim_params_dict, savepath)
print(f"vim and kl-matching params saved to {savepath}")

# metrics stats
savepath = os.path.join(save_dir, f"{filename}_train_stats{args.suffix}.pth")
torch.save(training_stats, savepath)
print(f"training metric stats saved to {savepath}")


