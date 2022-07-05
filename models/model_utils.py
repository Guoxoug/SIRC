# debugging
if __name__ == "__main__":
    from densenet import DenseNet
    from resnet import ResNet
    from mobilenet_v2 import MobileNetV2
    from resnet_v2 import ResNetV2
else:
    from models.densenet import DenseNet
    from models.resnet import ResNet, Bottleneck, conv1x1
    from models.mobilenet_v2 import MobileNetV2
    from models.resnet_v2 import ResNetV2

import re
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict, Optional

MODEL_TYPES = [
    "densenet", "resnet", "mobilenetv2", 
    "resnet50", "resnet101v2","densenet121"
]
MODEL_TYPE_MAPPINGS = {
    "densenet": DenseNet,
    "densenet121": DenseNet,
    "resnet":ResNet,
    "resnet50": ResNet,
    "resnet101v2": ResNetV2,
    "mobilenetv2": MobileNetV2,

}
MODEL_NAME_MAPPING = {
    "densenet121": "DenseNet-121",
    "resnet50": "ResNet-50",
    "mobilenetv2": "MobileNetV2",
    "resnet101v2": "ResNetV2-101"
}


def model_generator(model_type:str, **model_params) -> torch.nn.Module:
    """Construct a model following the supplied parameters."""
    assert model_type in MODEL_TYPES, (
        f"model type not supported"
        f"needs to be in {MODEL_TYPES}"    
    )

    # select model class
    Model = MODEL_TYPE_MAPPINGS[model_type]

    # override with proper values

    if model_type == "resnet50":
        model_params["layers"] = [3, 4, 6, 3]
        model_params["block"] = "bottleneck"


    # one off 
    # google's BiT implementation of ResNet is different to Torch's
    if model_type == "resnet101v2":
        model_params["block_units"] = [3, 4, 23, 3]
        model_params["width_factor"] = 1
    if model_type == "densenet121":
        model_params.update(
            {
            "growth_rate": 32,
            "block_config": [
                6,
                12,
                24,
                16
            ],
            "compression": 0.5,
            "num_init_features": 64,
            "bn_size": 4,
            "drop_rate": 0.0,
            "small_inputs": False
        }
        )

    # generic unpacking of pararmeters, need to match config file with 
    # model definition
    model = Model(**model_params)

    return model

def load_weights_from_file(
    model, weights_path, dev="cuda", keep_last_layer=True, 
):
    """Load parameters from a path of a file of a state_dict."""

    # special case for google BiTS model
    if type(model) == ResNetV2:
        model.load_from(np.load(weights_path), dev=dev)
        return 

    state_dict = torch.load(weights_path, map_location=dev)

    # special case for pretrained torchvision model
    # they fudged their original state dict and didn't change it
    if type(model) == DenseNet and "a639ec97" in weights_path: 
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        for key in list(state_dict.keys()):
            if "norm5" in key:
                new_key = key.replace("norm5", "norm_final")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict)
        return
   

    new_state_dict = OrderedDict()

    # data parallel trained models have module in state dict
    # prune this out of keys
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    # load params
    state_dict = new_state_dict
    if not keep_last_layer:

        # filter out final linear layer weights
        state_dict = {
            key: params for (key, params) in state_dict.items()
            if "classifier" not in key and "fc" not in key
        }
        model.load_state_dict(state_dict, strict=False)
    else:
        print("loading weights")
        model.load_state_dict(state_dict, strict=True)
       