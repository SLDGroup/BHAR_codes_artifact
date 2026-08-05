#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("tinyhar" "attend" "convlstm")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_classifier.py \
            --logging_prefix multisensor_classifier_window_25_acc \
            --architecture "$architecture" \
            --dataset dsads \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts torso right_arm left_arm right_leg left_leg \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
            --val_frac 0.1 \
            --window_size 25 \
            --overlap_frac 0.5 \
            --batch_size 256 \
            --lr 0.0001 \
            --epochs 25 \
            --ese 10 \
            --log_freq 200
    done
done

