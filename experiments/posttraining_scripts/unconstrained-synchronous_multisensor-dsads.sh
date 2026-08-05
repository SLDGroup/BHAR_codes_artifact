#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("tinyhar" "attend" "convlstm")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --multisensor_checkpoint_prefix multisensor_classifier_window_25_acc \
            --logging_prefix unconstrained-synchronous_multisensor \
            --policy unconstrained \
            --unconstrained_stride 12 \
            --model_type synchronous_multisensor \
            --architecture "$architecture" \
            --dataset dsads \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts torso right_arm left_arm right_leg left_leg \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
            --val_frac 0.1 \
            --window_size 25 \
            --harvesting_sensor_window_size 25 \
            --leakage 10e-6 \
            --sampling_frequency 25 \
            --max_energy 200e-6
    done
done

