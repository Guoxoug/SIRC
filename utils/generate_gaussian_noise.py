"""Generate noise images for OOD data."""
import torch
import torchvision as tv
import os
import PIL
from argparse import ArgumentParser
from numpy.random import randint, seed
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    "data_dir",
    type=str,
    help="directory to save generated dataset"
)
args = parser.parse_args()
if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)
save_dir = os.path.join(args.data_dir, "images")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
torch.manual_seed(0)
seed(0)

# random resolutions and standard deviations
res = randint(2, 256, size=10000)
std = torch.randn(10000)**2 
# mean and std
dataset = [
    torch.randn((3, res[i], res[i])) * std[i] + 0.5
    for i in tqdm(range(len(res)))
]
# to PIL multiplies by 255 for 3-bits and rounds
dataset = [tv.transforms.ToPILImage()(img) for img in dataset]
for i, datum in tqdm(enumerate(dataset)):
    savepath =  os.path.join(save_dir, f"img_{i}.jpg")
    datum = datum.resize((256,256), resample = PIL.Image.LANCZOS)
    datum.save(savepath)
