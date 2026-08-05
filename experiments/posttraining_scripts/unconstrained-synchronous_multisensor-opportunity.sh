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
            --dataset opportunity \
            --seed "$seed" \
            --subjects 1 2 3 4 \
            --sensors acc \
            --body_parts BACK RUA RLA LUA LLA L-SHOE R-SHOE \
            --activities 0 1 2 3 4 \
            --val_frac 0.1 \
            --window_size 25 \
            --harvesting_sensor_window_size 25 \
            --leakage 6.6e-6 \
            --sampling_frequency 25 \
            --max_energy 200e-6
    done
done

