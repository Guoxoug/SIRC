#!/bin/bash

while getopts t: flag
do
    case "${flag}" in
        t) test_only=${OPTARG};;
    esac
done


cd ..

# path the configuration file
config_path=experiment_configs/densenet121_imagenet200.json

# if not already defined
if [ -z ${test_only} ]; then
  test_only="0" 
  echo "training as well as testing"
fi

if [[ $test_only = "0" ]]; then
    echo "training"
    for num in $(seq 1 1 5)
    do
        python train.py $config_path --seed $num
    done
fi

for num in $(seq 1 1 5)
do
    python get_feature_quantiles.py $config_path --seed $num --gpu 0
    python get_gmm_params.py $config_path --seed $num --gpu 0
    python get_vim_params.py $config_path --seed $num --gpu 0
    python test.py $config_path --seed $num --gpu 0
done
