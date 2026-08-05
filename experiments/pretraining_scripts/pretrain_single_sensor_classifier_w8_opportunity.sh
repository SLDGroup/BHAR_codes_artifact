#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("tinyhar" "attend" "convlstm")
bodyparts=("BACK" "RUA" "RLA" "LUA" "LLA" "L-SHOE" "R-SHOE")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      for bp in "${bodyparts[@]}"; do
        python train_har_classifier.py \
              --logging_prefix "single_sensor_classifier_window_8_acc_${bp}" \
              --architecture "$architecture" \
              --dataset opportunity \
              --seed "$seed" \
              --subjects 1 2 3 4 \
              --sensors acc \
              --body_parts "$bp" \
              --activities 0 1 2 3 4 \
              --val_frac 0.1 \
              --window_size 8 \
              --overlap_frac 0.5 \
              --batch_size 256 \
              --lr 0.0001 \
              --epochs 25 \
              --ese 10 \
              --log_freq 200
      done
    done
done

