#!/bin/bash

pde=wave
seeds=(123 234 345 456 567 678 789 890)
losses=(mse)
n_neurons=(50 100 200 400 800)
n_layers=4
num_x=257
num_t=101
num_res=10000
opt=adam_lbfgs
switch_epochs=1000
adam_lrs=(0.00001 0.0001 0.001 0.01 0.1)
epochs=5000
betas=(1 2 3 4 5)
devices=(1 2 3 4 5 6 7)
proj=wave_adam_lbfgs_final
max_parallel_jobs=7

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

for seed in "${seeds[@]}"
do
    for loss in "${losses[@]}"
    do
        for n_neuron in "${n_neurons[@]}"
        do
            for beta in "${betas[@]}"
            do
                for adam_lr in "${adam_lrs[@]}"
                do
                    if [ $interrupted -eq 0 ]; then  # Check if Ctrl+C has been pressed
                        device=${devices[current_device]}
                        current_device=$(( (current_device + 1) % ${#devices[@]} ))

                        python run_experiment.py --seed $seed --pde $pde --pde_params beta $beta --opt $opt \
                            --opt_params switch_epochs $switch_epoch adam_lr $adam_lr lbfgs_history_size 100 --num_layers $n_layers --num_neurons $n_neuron \
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
                done
            done
        done
    done
done

# Wait for all background jobs to complete
wait

# Cleanup on normal exit
cleanup