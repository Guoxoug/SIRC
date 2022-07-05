import torch
import shutil
import os


OPTIMIZER_MAPPING = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop
}

SCHEDULER_MAPPING ={
    "multistep": torch.optim.lr_scheduler.MultiStepLR
}

# meters from https://github.com/pytorch/examples/blob/master/imagenet/main.py

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_filename(config: dict, seed=None) -> str:
    """Get the filename of a run (or a least the 1st part)."""

    # model type, dataset and random seed
    path = config["model"]["model_type"] + "_" + \
            config["id_dataset"]["name"] 
    if "early_exit_params" in config["model"]:
        path = path + "_" + "ee"
        if config["model"]["early_exit_params"]["frozen_backbone"]:
            path = path + "_frozen"
    if seed is not None:
        path = path + "_" + str(seed)

    return path 


def save_checkpoint(
    state, 
    is_best,
    config=None, 
    filepath=None
):
    """Save a training state as a checkpoint."""
    if config is None:
        pass
    elif filepath is None:
        filename = get_filename(config, seed=config["seed"]) + "_ckpt.pth"
        filepath = os.path.join(
            config["model"]["weights_path"], 
            filename
        )
    else:
        raise ValueError("You must supply either a path or a config to save.")

    torch.save(state, filepath)
    if is_best:
        filename = get_filename(config, seed=config["seed"]) + "_bestckpt.pth"
        shutil.copyfile(
            filepath,

            # TODO make this more general
            os.path.join(
                config["model"]["weights_path"],
                filename
            )
        )

def save_state_dict(model, config=None, filepath=None, is_best=False):
    """Save only the state dictionary of a model."""
    if config is None:
        pass
    elif filepath is None:
        filename = get_filename(config, seed=config["seed"]) + ".pth"
        filepath = os.path.join(
            config["model"]["weights_path"],
            filename
        )
    else:
        raise ValueError("You must supply either a path or a config to save.")

    torch.save(model.state_dict(), filepath)
    if is_best:
        filename = get_filename(config) + "_best.pth"
        shutil.copyfile(
            filepath,

            # TODO make this more general
            os.path.join(
                config["model"]["weights_path"],
                filename
            )
        )



