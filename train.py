from collections import OrderedDict
import os

import json
import time

from argparse import ArgumentParser

# argument parsing
parser = ArgumentParser()

parser.add_argument(
    "config_path",
    help="path to the experiment config file for this training script"
)

parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)

parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)

args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# set gpu
# MUST BE DONE BEFORE TORCH IS IMPORTED
if args.gpu is not None:
    config["gpu_id"] = args.gpu
elif "gpu_id" in config and (
    type(config["gpu_id"]) == int
    or 
    type(config["gpu_id"]) == list
):
    pass
else:
    config["gpu_id"] = 0

print(f"gpus allowed to be used: ", config["gpu_id"])
# will have brackets if read from json
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    config["gpu_id"]
).replace("[", "").replace("]", "")


import torch
from utils.data_utils import (
    Data, get_preprocessing_transforms, TRAIN_DATASETS
)
from utils.eval_utils import TopKError
from utils.train_utils import (
    OPTIMIZER_MAPPING, 
    SCHEDULER_MAPPING, 
    AverageMeter, 
    ProgressMeter,
    save_checkpoint,
    save_state_dict
)
from models.model_utils import  model_generator, load_weights_from_file

assert config["id_dataset"]["name"] in TRAIN_DATASETS, "not valid train set"


# set random seed
# CL arg overrides value in config file
if args.seed is not None:
    torch.manual_seed(args.seed)

    # add seed into config dictionary
    config["seed"] = args.seed

elif "seed" in config and type(config["seed"]) == int:
    torch.manual_seed(config['seed'])


# no seed in config or as CL arg
else:
    torch.manual_seed(0)
    config["seed"] = 0


# set training device, defaults to cuda
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dev = torch.device("cpu")
print(f"using {dev} for training")


# load training dataset
# more training arguments are passed directly from the configuration json
training_data = Data(
    **config["id_dataset"],
    transforms=get_preprocessing_transforms(config["id_dataset"]["name"])
)


# load the model
model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)


# training loss
# only standard one hot CE 

criterion = torch.nn.CrossEntropyLoss()

# pretrained weights if path supplied
if (
    "pretrained_path" in config["model"] 
    and 
    config["model"]["pretrained_path"] is not None
):
    if "keep_last_layer" in config["model"]:
        # where pretrained weights are
        load_weights_from_file(
            model, 
            config["model"]["pretrained_path"],
            keep_last_layer=config["model"]["keep_last_layer"]
        )
    else:
        load_weights_from_file(
            model,
            config["model"]["pretrained_path"],
            keep_last_layer=True
        )



# directory to save weights from training
if (
    "weights_path" in config["model"]
    and
    config["model"]["weights_path"] is not None
):
    # make a directory if it doesn't already exist
    if not os.path.exists(config["model"]["weights_path"]):
        os.mkdir(config["model"]["weights_path"])


# multigpu
if (
    config["data_parallel"] 
    and torch.cuda.device_count() > 1
    and dev.type == "cuda"
):
    model = torch.nn.DataParallel(model)
    multi_gpu = True
else:
    multi_gpu = False

model.to(dev)


# optimizer and scheduler
optimizer = OPTIMIZER_MAPPING[config["train_params"]["optimizer"]](
    model.parameters(), **config["train_params"]["optimizer_params"]
)
scheduler = SCHEDULER_MAPPING[config["train_params"]["lr_scheduler"]](
    optimizer, **config["train_params"]["lr_scheduler_params"]
)


def train_epoch(train_loader, model, criterion, optimizer, epoch:int):
    """Train the model for one epoch of the dataset."""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Err@1', ':6.2f')
    top5 = AverageMeter('Err@5', ':6.2f')
    ece = AverageMeter("ECE", ":6.2f")

    top1_calc = TopKError(k=1)
    top5_calc = TopKError(k=5)
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, ece],
        prefix=f"Epoch: [{epoch}]")

    # switch to train
    model.train()

    start = time.time()
    
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - start)

        # move data to correct device
        inputs, targets = inputs.to(dev), targets.to(dev)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

            
        # measure accuracy and record loss
        # note that outputs should be logits
        # targets should be labels (no distillation)
        err1 = top1_calc(targets, outputs)
        err5 = top5_calc(targets, outputs)
        batch_size = inputs.size(0) # may be smaller for last batch of epoch
        losses.update(loss.item(), batch_size)
        top1.update(err1, batch_size)
        top5.update(err5, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        # print every 5 batches
        if i % 5 == 0:
            progress.display(i)



def evaluate_epoch(val_loader, model, criterion, epoch: int) -> dict:
    """Evaluate the model for one epoch of the validation dataset."""
    batch_time = AverageMeter('Time', ':6.3f')

    meters = {
            "losses": AverageMeter('Loss', ':.4e'),
            "top1": AverageMeter('Err@1', ':6.2f'),
            "top5": AverageMeter('Err@5', ':6.2f'),
            "ece": AverageMeter("ECE", ":6.2f")
    }
    top1_calc = TopKError(k=1)
    top5_calc = TopKError(k=5)

    progress = ProgressMeter(
        len(val_loader),
        [
            batch_time, 
            meters["losses"], 
            meters["top1"],
            meters["top5"]
        ],
        prefix=f"Epoch: [{epoch}]")

    # switch to evaluation mode (e.g. freezes bn stats)
    model.eval()

    start = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):

            # move data to correct device
            inputs, targets = inputs.to(dev), targets.to(dev)

            outputs = model(inputs) # just logits no features
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            # note that outputs should be logits
            # targets should be labels (no distillation)
            err1 = top1_calc(targets, outputs)
            err5 = top5_calc(targets, outputs)
            batch_size = inputs.size(0) # may be smaller for last batch 
            meters["losses"].update(loss.item(), batch_size)
            meters["top1"].update(err1, batch_size)
            meters["top5"].update(err5, batch_size)


            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if i % 20 == 0:
                progress.display(i)
    
    eval_res = {}

    eval_res = {
        "err1": meters["top1"].avg,
        "err5": meters["top5"].avg,
        "ece": meters["ece"].avg,
        "loss": meters["losses"].avg
    }



    return eval_res

        
# training loop

for epoch in range(config["train_params"]["num_epochs"]):
    train_epoch(
        training_data.train_loader,
        model,
        criterion,
        optimizer,
        epoch
    )

    # reduce learning rate if at correct epoch
    scheduler.step()

    if training_data.val_size > 0:
        res = evaluate_epoch(
            training_data.val_loader,
            model,
            criterion,
            epoch
        )

    # TODO add loading from checkpoint
    save_checkpoint(
        {
            'epoch': epoch + 1,
            'config': args.config_path,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        },
        False,
        config
    )

# save at the end for future use
save_state_dict(model, config=config, is_best=False)



    



