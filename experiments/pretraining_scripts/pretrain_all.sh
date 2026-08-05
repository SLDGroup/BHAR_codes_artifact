#!/bin/bash

# Name of the tmux session
SESSION="pretrain"

# List of window names (and corresponding script identifiers)
DATASETS=("dsads" "rwhar" "pamap2" "opportunity")

# Start a new detached tmux session with the first window
tmux new-session -d -s "$SESSION" -n "${DATASETS[0]}"

# Enable mouse mode for mouse scrolling, clicking windows, and resizing
tmux set-option -t "$SESSION" mouse on

# Run the first script in the first window
tmux send-keys -t "$SESSION:${DATASETS[0]}" "./pretrain_${DATASETS[0]}.sh" C-m

# Loop through the remaining datasets to create windows and run their scripts
for ((i=1; i<${#DATASETS[@]}; i++)); do
    DATASET="${DATASETS[i]}"
    
    # Create a new window with the dataset name
    tmux new-window -t "$SESSION" -n "$DATASET"
    
    # Send the script execution command to the newly created window
    tmux send-keys -t "$SESSION:$DATASET" "./pretrain_${DATASET}.sh" C-m
done

# Select the first window as active
tmux select-window -t "$SESSION:${DATASETS[0]}"

# Attach to the tmux session
tmux attach-session -t "$SESSION"