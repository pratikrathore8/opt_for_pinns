# Challenges in Training PINNs: A Loss Landscape Perspective

This repository contains companion code for "Challenges in Training PINNs: A Loss Landscape Perspective". We provide instructions for reproducing our experiments and plots. 

## Installation

After the repository is downloaded or cloned, please change to the `opt_for_pinns/` directory and run the following code to pip install packages specified in `requirements.txt`: 

```
pip install -r requirements.txt
```

## Instructions

To run a single experiment, one can run `run_experiment.py` with arguments: 

```
python run_experiment.py [--seed] [--pde] [--pde_params] [--opt] [--opt_params] [--num_layers] [--num_neurons] [--loss] [--num_x] [--num_t] [--num_res] [--epochs] [--wandb_project] [--device]
```

- `seed`: initial seed for reproducibility
- `pde`: name of the PDE type
- `pde_params`: PDE coefficients
- `opt`: name of the optimizer (could also be an optimizer combo e.g. Adam + L-BFGS)
- `opt_params`: optimizer parameters
- `num_layers`: number of layers of the PINN neural network
- `num_neurons`: number of neurons of each hidden layer
- `loss`: name of the loss function used for each component of the PINN loss
- `num_x`: number of spatial points for the grid
- `num_t`: number of temporal points for the grid
- `num_res`: number of residual points to sample from the grid
- `epochs`: total number of epochs to optimize the model for
- `wandb_project`: name for the Weights & Biases project (for logging and monitoring)
- `device`: identifier of the GPU to be used for training

All of the experiment results discussed in the paper can be reproduced by running shell scripts located in the `config/` sub-directory. For example, to run the experiments for the convection problem using NysNewton-CG (NNCG) after Adam + L-BFGS (discussed in section 7.3), one can run: 

```
sh config/convection_adam_lbfgs_nncg_best.sh
```

Notebooks in the `plotting/` sub-directory can be used to reproduce the plots shown in the paper. 
