#!/bin/bash

pde=convection
seeds=(123 234 345 456 567 678 789 890)
losses=(mse huber)
n_neurons=(50 100 200)
n_layers=4
opt=lbfgs
epochs=1000
betas=(1 10 20 30 40)
devices=(0 3 4 5 6 7)
proj=convection_lbfgs_v2
max_parallel_jobs=6

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
                if [ $interrupted -eq 0 ]; then  # Check if Ctrl+C has been pressed
                    device=${devices[current_device]}
                    current_device=$(( (current_device + 1) % ${#devices[@]} ))

                    python run_experiment.py --seed $seed --pde $pde --pde_params beta $beta --opt $opt \
                        --opt_params history_size 100 --num_layers $n_layers --num_neurons $n_neuron \
                        --loss $loss --num_x 101 --num_t 101 --epochs $epochs --wandb_project $proj \
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

# Wait for all background jobs to complete
wait

# Cleanup on normal exit
cleanup

# pde=convection
# # seeds=(0 1 2 3 4 5 6 7 8 9)
# # losses=(mse huber l1)
# # n_neurons=(25 50 100 200)
# seeds=(0)
# # seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)
# # losses=(mse huber hybrid)
# losses=(mse)
# n_neurons=(100)
# n_layers=4
# opt=lbfgs
# epochs=1000
# # betas=(1 10 20 30 40)
# # betas=(30)
# betas=(30)
# # devices=(0 1 2 3 4 5 6 7)
# devices=(0 3 4 5 6 7)
# proj=convection_lbfgs
# # max_parallel_jobs=8  # Set this to the number of GPUs or a safe number of parallel jobs
# max_parallel_jobs=6  # Set this to the number of GPUs or a safe number of parallel jobs

# # Initialize current device index
# current_device=0

# for seed in ${seeds[@]}
# do
#     for loss in ${losses[@]}
#     do
#         for n_neurons in ${n_neurons[@]}
#         do
#             for beta in ${betas[@]}
#             do
#                 device=${devices[current_device]}
#                 current_device=$(( (current_device + 1) % ${#devices[@]} ))

#                 python run_experiment.py --seed $seed --pde $pde --pde_params beta $beta --opt $opt \
#                     --opt_params history_size 100 --num_layers $n_layers --num_neurons $n_neurons \
#                     --loss $loss --num_x 101 --num_t 101 --epochs $epochs --wandb_project $proj \
#                     --device $device &

#                 # Limit the number of parallel jobs
#                 [ $(jobs | wc -l) -ge $max_parallel_jobs ] && wait -n
#             done
#         done
#     done
# done

# wait