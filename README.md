# Challenges in Training PINNs: A Loss Landscape Perspective

This repository contains companion code for "[Challenges in Training PINNs: A Loss Landscape Perspective](https://arxiv.org/abs/2402.01868)". We provide instructions for reproducing our experiments and plots. 

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

## Reproducing results

One can run shell scripts (`.sh`) in the `config` folder to reproduce experiment results. For example, to run the experiments for the convection problem using NysNewton-CG (NNCG) after Adam + L-BFGS (discussed in section 7.3), one can run: 

```
sh config/convection_adam_lbfgs_nncg_best.sh
```

We detail instructions for reproducing each plot in the paper below. 

### Quality solution and low loss (Section 4) 

First, please run: 

- `convection_adam.sh`
- `convection_lbfgs.sh`
- `convection_adam_lbfgs.sh`
- `convection_adam_lbfgs_adj.sh`
- `convection_adam_lbfgs_adj2.sh`
- `reaction_adam.sh`
- `reaction_lbfgs.sh`
- `reaction_adam_lbfgs.sh`
- `reaction_adam_lbfgs_adj.sh`
- `reaction_adam_lbfgs_adj2.sh`
- `wave_adam.sh`
- `wave_lbfgs.sh`
- `wave_adam_lbfgs.sh`
- `wave_adam_lbfgs_adj.sh`
- `wave_adam_lbfgs_adj2.sh`

Once these scripts are finished running, please run `l2re_loss_scatter.ipynb` to generate Figure 2. Note that `entity_name` in the notebook needs to be correctly specified to the associated Weights & Biases account. 

### Ill-conditioning (Section 5)

Please run `spectral_densities.ipynb` to generate Figure 3 and Figure 7. 

### Performance of Adam + L-BFGS (Section 6 & Appendix D)

First, please run (if not already): 

- `convection_adam.sh`
- `convection_lbfgs.sh`
- `convection_adam_lbfgs.sh`
- `convection_adam_lbfgs_adj.sh`
- `convection_adam_lbfgs_adj2.sh`
- `reaction_adam.sh`
- `reaction_lbfgs.sh`
- `reaction_adam_lbfgs.sh`
- `reaction_adam_lbfgs_adj.sh`
- `reaction_adam_lbfgs_adj2.sh`
- `wave_adam.sh`
- `wave_lbfgs.sh`
- `wave_adam_lbfgs.sh`
- `wave_adam_lbfgs_adj.sh`
- `wave_adam_lbfgs_adj2.sh`

Once these scripts are finished running, please run `opt_comparison.ipynb` to generate Figure 8. Note that `entity_name` in the notebook needs to be correctly specified to the associated Weights & Biases account. 

### Under-optimization (Section 7 & Appendix E)

First, please run: 

- `convection_adam_lbfgs_nncg_best.sh`
- `convection_adam_lbfgs_gd_best.sh`
- `reaction_adam_lbfgs_nncg_best.sh`
- `reaction_adam_lbfgs_gd_best.sh`
- `wave_adam_lbfgs_nncg_best.sh`
- `wave_adam_lbfgs_gd_best.sh`

Once these scripts are finished running, please run `under_optimization.ipynb` to generate Figure 8 and Figure 4. Note that `entity_name` in the notebook needs to be correctly specified to the associated Weights & Biases account. 

Please run `line_search.ipynb` to generate Figure 9.

### Solution visualization (Section 7 & Appendix B)

Please run `solution_visualizations.ipynb` to generate Figure 5 and Figure 6. 

### Condition number and the number of residual points (Appendix F)

Please run `condition_number.ipynb` to generate Figure 10.

## Citation

If you found this repository to be useful in your work, please cite the following paper: 

```
@inproceedings{rathore2024challenges,
  title = {Challenges in Training {PINN}s: A Loss Landscape Perspective},
  author = {Rathore, Pratik and Lei, Weimu and Frangella, Zachary and Lu, Lu and Udell, Madeleine},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024},
}
```
