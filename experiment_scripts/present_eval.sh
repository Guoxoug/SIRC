#!/bin/bash
cd ..

# plot sequentially over each architecture/dataset combination
config_path=experiment_configs/mobilenetv2_imagenet200.json
python plot_vary_sc_params.py  $config_path --num_runs 5 
python plot_ood_sc_comp.py $config_path 5


config_path=experiment_configs/resnet50_imagenet200.json
python plot_vary_sc_params.py  $config_path --num_runs 5  
python plot_ood_sc_comp.py $config_path 5


config_path=experiment_configs/densenet121_imagenet200.json
python plot_vary_sc_params.py  $config_path --num_runs 5 
python plot_ood_sc_comp.py $config_path 5

config_path=experiment_configs/densenet121_imagenet.json
python plot_ood_sc_comp.py $config_path 1 --seed 0

config_path=experiment_configs/resnet101v2_imagenet.json
python plot_ood_sc_comp.py $config_path 1 --seed 0

# these aggregate over a number of architectures
python plot_id_ood_2d_full.py imagenet200 inaturalist --seed 1
python plot_id_ood_2d_full.py imagenet200 textures --seed 1
python plot_id_ood_2d_full.py imagenet inaturalist --seed 0
python plot_id_ood_2d_full.py imagenet textures --seed 0

# print tables to command line
config_path=experiment_configs/mobilenetv2_imagenet200.json
python table.py $config_path 5 --latex 0
config_path=experiment_configs/resnet50_imagenet200.json
python table.py  $config_path 5  --latex 0
config_path=experiment_configs/densenet121_imagenet200.json
python table.py  $config_path 5 --latex 0
config_path=experiment_configs/densenet121_imagenet.json
python table.py $config_path 1 --seed 0 --latex 0
config_path=experiment_configs/resnet101v2_imagenet.json
python table.py $config_path 1 --seed 0 --latex 0


