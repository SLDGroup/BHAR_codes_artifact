#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("tinyhar" "attend" "convlstm")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_single_sensor \
            --policy opportunistic \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset opportunity \
            --seed "$seed" \
            --subjects 1 2 3 4 \
            --sensors acc \
            --body_parts BACK RUA RLA LUA LLA L-SHOE R-SHOE \
            --activities 0 1 2 3 4 \
            --val_frac 0.1 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --policy_frequency 5 \
            --init_energy 0 \
            --sampling_frequency 25 \
            --max_energy 200e-6
    done
done

