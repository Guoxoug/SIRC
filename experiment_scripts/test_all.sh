#!/bin/bash
cd ..

# NOTE
# uncomment and run eval_logits_features.py 
# instead of test.py if logits and features are already 
# generated and saved

# gpu command line arg may not work
# use 
# export CUDA_VISIBLE_DEVICES=<gpu_id>
# if that is the case

config=experiment_configs/resnet101v2_imagenet.json
weights=models/saved_models/BiT-S-R101x1.npz

python get_gmm_params.py $config \
    --seed 0 --gpu 0 \
    --weights_path $weights

python get_vim_sirc_params.py $config \
    --seed 0 --gpu 0 \
    --weights_path $weights

python test.py $config \
    --seed 0 --gpu 0 \
    --weights_path $weights

# python eval_logits_features.py $config \
#     --seed 0 --gpu 0 

config=experiment_configs/densenet121_imagenet.json
weights=models/saved_models/densenet121-a639ec97.pth

python get_gmm_params.py $config \
    --seed 0 --gpu 0 \
    --weights_path $weights

python get_vim_sirc_params.py $config \
    --seed 0 --gpu 0 \
    --weights_path $weights

python test.py $config \
    --seed 0 --gpu 0 \
    --weights_path $weights

# python eval_logits_features.py $config \
#     --seed 0 --gpu 0

for num in $(seq 1 1 5)
do

python get_vim_sirc_params.py experiment_configs/resnet50_imagenet200.json --seed $num --gpu 0  
python get_vim_sirc_params.py experiment_configs/mobilenetv2_imagenet200.json --seed $num --gpu 0 
python get_vim_sirc_params.py experiment_configs/densenet121_imagenet200.json --seed $num --gpu 0 

python get_gmm_params.py experiment_configs/resnet50_imagenet200.json --seed $num --gpu 0  
python get_gmm_params.py experiment_configs/mobilenetv2_imagenet200.json --seed $num --gpu 0
python get_gmm_params.py experiment_configs/densenet121_imagenet200.json --seed $num --gpu 0

python test.py experiment_configs/resnet50_imagenet200.json --seed $num --gpu 0
python test.py experiment_configs/mobilenetv2_imagenet200.json --seed $num --gpu 0
python test.py experiment_configs/densenet121_imagenet200.json --seed $num --gpu 0

# python eval_logits_features.py experiment_configs/resnet50_imagenet200.json --seed $num --gpu 0
# python eval_logits_features.py experiment_configs/mobilenetv2_imagenet200.json --seed $num --gpu 0
# python eval_logits_features.py experiment_configs/densenet121_imagenet200.json --seed $num --gpu 0

done

echo "finished testing"

