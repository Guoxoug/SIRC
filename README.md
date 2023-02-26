# Augmenting Softmax Information for Selective Classification with Out-of-Distribution Data

This repository contains code for our paper that received the best paper award at ACCV 2022.
If you found either the paper or the code useful please consider citing it:
```bibtex
@InProceedings{Xia_2022_ACCV,
    author    = {Xia, Guoxuan and Bouganis, Christos-Savvas},
    title     = {Augmenting Softmax Information for Selective Classification with Out-of-Distribution Data},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {1995-2012}
}
```

## Requirements
The main requirements for this repository are:
```
python
pytorch (+torchvision)
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
opencv
pillow
```


## Datasets and Setup
The datasets used in our work can be obtained by following the links and/or instructions below.
- [ImageNet-200, Near-ImageNet-200 and Caltech-45](https://github.com/daintlab/unknown-detection-benchmarks): download the [ImageNet-based benchmark](https://docs.google.com/uc?export=download&id=1gapHov_B-DZ9bKOffg2DFx7lLPOe1T7l)
- [ImageNet](https://www.image-net.org/)
- [OpenImage-O](https://github.com/haoqiwang/vim): download the [test set](https://github.com/cvdfoundation/open-images-dataset) and place the datalist file `utils/openimages_datalist.txt` the level above the directory containing the images.
- [iNaturalist](https://github.com/deeplearning-wisc/large_scale_ood)
- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/): download the dataset and place `utils/textures_datalist.txt` the level above the directory containing the images.
- [Colonoscopy](http://www.depeca.uah.es/colonoscopy_dataset/): run `python utils/extract_frames.py <path/to/data_dir>` to download the videos and extract their frames to the specified directory.
- [Colorectal](https://zenodo.org/record/53169#.Yr21hXbMJ3j)
- Noise: run `python utils/generate_gaussian_noise.py <path/to/data_dir>` to generate the data and save it to the specified directory.
- [ImageNet-O](https://github.com/hendrycks/natural-adv-examples)

After obtaining the data edit `experiment_configs/change_paths.py` such that the dictionary `data_paths` is updated with the correct paths to all the datasets and `RESULTS` points to the directory where you want results (plots and `.csv` files) to be saved. Then run the script to update all the configuration `.json` files.
```bash
cd experiment_configs
python change_paths.py
cd ..
```
## Training
To train and test 5 models from scratch for an architecture run:
```bash
cd models
mkdir saved_models
cd ..
cd experiment_scripts
chmod +x *
./<network>_imagenet200.sh
cd ..
``` 
You can specify which GPU to use within the files. The training script allows multigpu, but this must be specified in the `experiment_config/<network>_imagenet200.json` configuration file rather than a command line argument. For example, you could edit the field `"gpu_id": [0,1,2]` to use 3 GPUs.
## Testing
We also provide the [weights](https://drive.google.com/uc?export=download&id=1MoVCligDFmnN84GxOF4tJ9MYNOyql8ra
) for our trained models as well as the pretrained [DenseNet-121](https://download.pytorch.org/models/densenet121-a639ec97.pth) and [ResNetV2-101](https://storage.googleapis.com/bit_models/BiT-S-R101x1.npz) (which are also available via their respective links). Once you have `saved_models.zip`, place it in `models/` and `unzip` it.
```bash
cd models
unzip saved_models.zip
cd ..
```
To test the trained models run 
```bash
cd experiment_scripts
chmod +x *
./test_all.sh
cd ..
```
`get_gmm_params.py`, `get_vim_sirc_params.py` and `test.py` run inference and save parameters and logits/features respectively, so they only need to be run once. If you want to re-run evaluation then you can comment out the above scripts from `test_all.sh` and just run `eval_logits_features.py` to obtain results `.csv` files. 

To obtain tables and plots from our paper after testing all networks run 
```bash
cd experiment_scripts
./present_eval.sh
cd ..
```
 The `--latex` command line argument for `table.py` controls whether the tables are printed in TeX or not. Set it to `1` inside `present_eval.sh` if you want to render the tables like in the paper.
