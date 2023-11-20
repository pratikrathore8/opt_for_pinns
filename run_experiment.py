# external libraries and packages
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import wandb
import argparse

def main(): 
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=1234, help='initial seed')
  parser.add_argument('--pde', type=str, default='convection', help='PDE type')
  parser.add_argument('--opt', type=str, default='lbfgs', help='optimizer to use')
  parser.add_argument('--num_layers', type=int, default=4, help='number of layers of the neural net')
  parser.add_argument('--num_neurons', type=int, default=50, help='number of neurons per layer')
  parser.add_argument('--loss', type=str, default='mse', help='type of loss function')
  parser.add_argument('--num_x', type=int, default=101, help='')
  parser.add_argument('--num_t', type=int, default=101, help='')

  parser.add_argument('--batch_size', type=int, default=None) # Number of samples in each batch

  parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to run')
  parser.add_argument('--log_frequency', type=int, default=100, help='number of epochs between logging actions')
  parser.add_argument('--wandb_project', type=str, default='PINNs', help='W&B project name')
  parser.add_argument('--device', type=str, default=0, help='GPU to use')
  parser.add_argument('--params', nargs='+', type=str, default='learning_rate')

  # Extract arguments from parser
  args = parser.parse_args()
  # set initial seed
  initial_seed = args.seed
  set_random_seed(initial_seed)
  # select PDE to solve
  pde_name, pde_selection = get_valid_pde(args.pde)
  # obtain spatial and temporal domains of the PDE
  domain_x, domain_t = get_pde_domain(pde_selection)
  # obtain PDE specific parameters
  pde_params = get_pde_params(pde_selection)
  # select model architecture
  model_name, model_selection = get_valid_model(args.model)
  # select optimizer
  optimizer_name, optimizer_selection = get_valid_optimizer(args.opt)
  # specify parameters for the PINNsFormer
  step_size = args.step_size
  num_steps = args.num_steps
  # specify size of the training set
  num_samples_x, num_samples_t = get_num_samples(args.num_samples)
  # specify training length
  num_epochs = args.epochs
  # specify training batch size
  batch_size = args.batch_size
  # specify the logging frequency for Weights and Biases
  log_frequency = args.log_frequency
  # specify Weights and Biases project to log to
  wandb_project = args.wandb_project
  # specify GPU to use
  device = "cuda:" + args.device

  # organize arguments for the experiment into a dictionary for logging purpose
  experiment_args = {
    "pde": pde_name, 
    "initial_seed": initial_seed, 
    "model_name": model_name,
    "optimizer_name": optimizer_name, 
    "num_samples_x": num_samples_x, 
    "num_samples_t": num_samples_t, 
    "step_size": step_size,
    "num_steps": num_steps,
    "num_epochs": num_epochs, 
    "batch_size": batch_size,
    "log_frequency": log_frequency, 
    "wandb_project": wandb_project,
    "device": device
  }

  # print out arguments
  print("Seed set to: {}".format(initial_seed))
  print("Selected PDE type: {}".format(pde_name))
  print("Selected model: {}".format(model_name))
  print("Optimizer to use: {}".format(optimizer_name))
  print("Step size for PINNsFormer: {}".format(step_size))
  print("Number of steps for PINNsFormer: {}".format(num_steps))
  print("Number of spatial points (x): {}".format(num_samples_x))
  print("Number of temporal points (t): {}".format(num_samples_t))
  print("Number of epochs: {}".format(num_epochs))
  print("Batch size: {}".format(batch_size))
  print("Logging frequency: {}".format(log_frequency))
  print("Weights and Biases project: {}".format(wandb_project))
  print("GPU to use: {}".format(device))

  run_name = f"{pde_name}/{model_name}/{optimizer_name}"
  with wandb.init(project=wandb_project, name=run_name, config=experiment_args):
    # add more information to W&B config if necessary
    update_wandb_config(wandb, optimizer_name)
    # generate training points
    training_data = get_data(domain_x, domain_t, num_samples_x, num_samples_t)
    # generate test points and corresponding solutions
    test_points, _, _, _, _ = get_data(domain_x, domain_t, 101, 101)
    ground_truths = get_solution(test_points, pde_selection, pde_params)
    test_data = (test_points, ground_truths)
    # obtain optimizer parameters
    opt_params = get_opt_params(optimizer_selection, {})
    # train the model
    try: 
      train(training_data=training_data, 
        test_data=test_data, 
        pde=pde_selection,
        pde_params=pde_params,
        opt=optimizer_selection,
        opt_params=opt_params,
        model_architecture=model_selection,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_steps=num_steps,
        step_size=step_size,
        log_frequency=log_frequency,
        device=device
        )
    # log error and traceback info to W&B, and exit gracefully
    except Exception as e:
      traceback.print_exc(file=sys.stderr)
      raise e

if __name__ == "__main__":
  main()