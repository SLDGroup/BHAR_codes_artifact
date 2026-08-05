#!/bin/bash
cd ..

seeds=(0 1 2)
architectures=("tinyhar")

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_single_sensor_ablation_dur_lower \
            --policy opportunistic \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset pamap2 \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts hand chest ankle \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
            --val_frac 0.2 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --policy_frequency 5 \
            --init_energy 0 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --activity_duration 0 10
    done
done

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_single_sensor_ablation_dur_higher \
            --policy opportunistic \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset pamap2 \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts hand chest ankle \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
            --val_frac 0.2 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --policy_frequency 5 \
            --init_energy 0 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --activity_duration 30 40
    done
done




for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_single_sensor_ablation_thresh_lower \
            --policy opportunistic \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset pamap2 \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts hand chest ankle \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
            --val_frac 0.2 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --policy_frequency 5 \
            --init_energy 0 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --tx_thresh 25e-6
    done
done

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_single_sensor_ablation_thresh_higher \
            --policy opportunistic \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset pamap2 \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts hand chest ankle \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
            --val_frac 0.2 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --policy_frequency 5 \
            --init_energy 0 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --tx_thresh 100e-6
    done
done

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_single_sensor_ablation_leakage_lower \
            --policy opportunistic \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset pamap2 \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts hand chest ankle \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
            --val_frac 0.2 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 3.3e-6 \
            --policy_frequency 5 \
            --init_energy 0 \
            --sampling_frequency 25 \
            --max_energy 200e-6
    done
done

for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_single_sensor_ablation_leakage_higher \
            --policy opportunistic \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset pamap2 \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts hand chest ankle \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
            --val_frac 0.2 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 13.2e-6 \
            --policy_frequency 5 \
            --init_energy 0 \
            --sampling_frequency 25 \
            --max_energy 200e-6
    done
done




for seed in "${seeds[@]}"; do
    for architecture in "${architectures[@]}"; do
      python train_har_policy.py \
            --single_sensor_checkpoint_prefix single_sensor_classifier_window_8_acc \
            --logging_prefix opportunistic-asynchronous_single_sensor_ablation_efficiency_lower \
            --policy opportunistic \
            --model_type asynchronous_single_sensor \
            --architecture "$architecture" \
            --dataset pamap2 \
            --seed "$seed" \
            --subjects 1 2 3 4 5 6 7 8 \
            --sensors acc \
            --body_parts hand chest ankle \
            --activities 0 1 2 3 4 5 6 7 8 9 10 11 \
            --val_frac 0.2 \
            --window_size 8 \
            --harvesting_sensor_window_size 8 \
            --leakage 6.6e-6 \
            --policy_frequency 5 \
            --init_energy 0 \
            --sampling_frequency 25 \
            --max_energy 200e-6 \
            --harvest_efficiency 0.25
    done
done