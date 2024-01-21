#!/bin/bash

pde=convection
seed=345
loss=mse
n_neuron=200
n_layers=4
num_x=257
num_t=101
num_res=10000
opt=adam_lbfgs_gd
switch_epoch_lbfgs=11000
switch_epoch_gd=12500
adam_lr=0.0001
epochs=14500
beta=40
devices=(3)
proj=convection_adam_lbfgs_gd_best
max_parallel_jobs=1

background_pids=()
current_device=0

interrupted=0  # Flag to indicate if Ctrl+C is pressed

# Function to handle SIGINT (Ctrl+C)
cleanup() {
    echo "Interrupt received, stopping background jobs..."
    interrupted=1  # Set the flag
    for pid in "${background_pids[@]}"; do
        kill $pid 2>/dev/null
    done
}

# Trap SIGINT
trap cleanup SIGINT

if [ $interrupted -eq 0 ]; then  # Check if Ctrl+C has been pressed
    device=${devices[current_device]}
    current_device=$(( (current_device + 1) % ${#devices[@]} ))

    python run_experiment.py --seed $seed --pde $pde --pde_params beta $beta --opt $opt \
        --opt_params adam_lr $adam_lr lbfgs_history_size 100 switch_epoch_lbfgs $switch_epoch_lbfgs switch_epoch_gd $switch_epoch_gd --num_layers $n_layers --num_neurons $n_neuron \
        --loss $loss --num_x $num_x --num_t $num_t --num_res $num_res --epochs $epochs --wandb_project $proj \
        --device $device &

    background_pids+=($!)

    # Limit the number of parallel jobs
    while [ $(jobs | wc -l) -ge $max_parallel_jobs ]; do
        wait -n
        # Clean up finished jobs from the list
        for i in ${!background_pids[@]}; do
            if ! kill -0 ${background_pids[$i]} 2> /dev/null; then
                unset 'background_pids[$i]'
            fi
        done
    done
fi