import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
import random
import re
import wandb
import os
import shutil
import bz2
import pickle

"""
Helper function for obtaining corresponding domain and loss function of the chosen PDE type. 

INPUT: 
- pde_name: string; name of the PDE problem
- pde_params_list: list of strings; coefficients of the PDE
- loss_name: string; name of the loss type
OUTPUT: 
- x_range: list of size 2; lower and upper bounds of spatial variable x
- t_range: list of size 2; lower and upper bounds of temporal variable t
- loss_func: loss function that takes (x,t,pred) and computes the total loss
- pde_coefs: dictionary containing coefficients of the PDE
"""
def get_pde(pde_name, pde_params_list, loss_name): 
    # determine loss type
    loss_options = {
        "l1": {"res": nn.L1Loss(), "bc": nn.L1Loss(), "ic": nn.L1Loss()},
        "mse": {"res": nn.MSELoss(), "bc": nn.MSELoss(), "ic": nn.MSELoss()},
        "huber": {"res": nn.HuberLoss(), "bc": nn.HuberLoss(), "ic": nn.HuberLoss()},
        "hybrid": {"res": nn.HuberLoss(), "bc": nn.MSELoss(), "ic": nn.MSELoss()}
    }
    try: 
        loss_type = loss_options[loss_name]
    except KeyError as ke:
        raise RuntimeError("{} is not a valid loss type.".format(ke))

    # parse PDE parameters
    pde_coefs = parse_params_list(pde_params_list)
    
    # determine pde type
    if pde_name == "convection": 
        if "beta" not in pde_coefs.keys(): 
            raise KeyError("beta is not specified for convection PDE.")

        x_range = [0, 2 * np.pi]
        t_range = [0, 1]

        def loss_func(x, t, pred): 
            x_res, x_left, x_right, x_upper, x_lower = x
            t_res, t_left, t_right, t_upper, t_lower = t
            outputs_res, outputs_left, outputs_right, outputs_upper, outputs_lower = pred

            u_x = torch.autograd.grad(outputs_res, x_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]
            u_t = torch.autograd.grad(outputs_res, t_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]

            loss_res = loss_type["res"](u_t + pde_coefs["beta"] * u_x, torch.zeros_like(u_t))
            loss_bc = loss_type["bc"](outputs_upper - outputs_lower, torch.zeros_like(outputs_upper))
            loss_ic = loss_type["ic"](outputs_left[:,0], torch.sin(x_left[:,0]))

            loss = loss_res + loss_bc + loss_ic

            return loss
    
    else: 
        raise RuntimeError("{} is not a valid PDE name.".format(pde_name))

    return x_range, t_range, loss_func, pde_coefs


def get_ref_solutions(pde_name, pde_coefs, x, t): 
    if pde_name == "convection": 
        sol = np.sin(x[0].cpu().detach().numpy() - pde_coefs["beta"] * t[0].cpu().detach().numpy())
    else: 
        raise RuntimeError("{} is not a valid PDE name.".format(pde_name))
    
    return sol

"""
Helper function for setting seed for the random number generator in various packages.

INPUT: 
- seed: integer
"""
def set_random_seed(seed): 
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

"""
Helper function for generating data on a grid. 
Adapted from implementation: https://github.com/AdityaLab/pinnsformer/blob/main/util.py

INPUT: 
- x_range: list of size 2; lower and upper bounds of spatial variable x
- t_range: list of size 2; lower and upper bounds of temporal variable t
- x_num: positive integer; number of x points
- t_num: positive integer; number of t points
- random: boolean; indication whether to (uniformly) randomly from the grid
- device: string; the device that the samples will be stored at
OUTPUT: 
- x: tuple of (x_res, x_left, x_right, x_upper, x_lower)
- t: tuple of (t_res, t_left, t_right, t_upper, t_lower)
where: 
> res: numpy array / tensor of size (x_num * t_num) * 2; residual points -- all of the grid points
> b_left: numpy array / tensor of size (x_num) * 2; initial points (corresponding to initial time step)
> b_right: numpy array / tensor of size (x_num) * 2; terminal points (corresponding to terminal time step)
> b_upper: numpy array / tensor of size (t_num) * 2; upper boundary points
> b_lower: numpy array / tensor of size (t_num) * 2; lower boundary points
"""
def get_data(x_range, t_range, x_num, t_num, random=False, device='cpu'):
  if random: 
    x = np.concatenate(([x_range[0]], np.random.uniform(x_range[0], x_range[1], x_num-2), [x_range[1]]))
    t = np.concatenate(([t_range[0]], np.random.uniform(t_range[0], t_range[1], t_num-2), [t_range[1]]))
  else: 
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(t_range[0], t_range[1], t_num)

  x_mesh, t_mesh = np.meshgrid(x,t)
  data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)

  b_left = data[0,:,:] 
  b_right = data[-1,:,:]
  b_upper = data[:,-1,:]
  b_lower = data[:,0,:]
  res = data.reshape(-1,2)

  if device != 'cpu': 
    res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
    b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
    b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
    b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
    b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

  x_res, t_res = res[:,0:1], res[:,1:2]
  x_left, t_left = b_left[:,0:1], b_left[:,1:2]
  x_right, t_right = b_right[:,0:1], b_right[:,1:2]
  x_upper, t_upper = b_upper[:,0:1], b_upper[:,1:2]
  x_lower, t_lower = b_lower[:,0:1], b_lower[:,1:2]

  x = (x_res, x_left, x_right, x_upper, x_lower)
  t = (t_res, t_left, t_right, t_upper, t_lower)

  return x, t

"""
Helper function for initializing neural net parameters. 
"""
def init_weights(m):
  if isinstance(m, nn.Linear):
    # torch.nn.init.xavier_uniform_(m.weight)
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.0)

"""
Helper function for making predictions with PINN. 

INPUT: 
- x: tutple of (x_res, x_left, x_right, x_upper, x_lower)
- t: tutple of (t_res, t_left, t_right, t_upper, t_lower)
- model: PINN model
OUTPUT: 
- preds: tuple of (pred_res, pred_left, pred_right, pred_upper, pred_lower)
where: 
> pred_res: predictions on residual points
> pred_left: predictions on initial points
> pred_right: predictions on terminal points
> pred_upper: predictions on upper boundary points
> pred_lower: predictions on lower boundary points
"""
def predict(x, t, model): 

    x_res, x_left, x_right, x_upper, x_lower = x
    t_res, t_left, t_right, t_upper, t_lower = t
    
    pred_res = model(x_res, t_res)
    pred_left = model(x_left, t_left)
    pred_right = model(x_right, t_right)
    pred_upper = model(x_upper, t_upper)
    pred_lower = model(x_lower, t_lower)
    
    preds = (pred_res, pred_left, pred_right, pred_upper, pred_lower) 
    
    return preds

"""
Helper function for computing l1 relative error. 

INPUT: 
- prediction: numpy array of predictions from the model
- target: numpy array of ground truths
OUTPUT: 
- error: scalar; computed relative error
"""
def l1_relative_error(prediction, target): 
    return np.sum(np.abs(target-prediction)) / np.sum(np.abs(target))

"""
Helper function for computing l2 relative error. 

INPUT: 
- prediction: numpy array of predictions from the model
- target: numpy array of ground truths
OUTPUT: 
- error: scalar; computed relative error
"""
def l2_relative_error(prediction, target): 
    return np.sqrt(np.sum((target-prediction)**2) / np.sum(target**2))

def get_opt(opt_name, opt_params, model_params):
    if opt_name == 'adam':
        return Adam(model_params, **opt_params)
    elif opt_name == 'lbfgs':
        if "history_size" in opt_params:
            opt_params["history_size"] = int(opt_params["history_size"])
        return LBFGS(model_params, **opt_params, line_search_fn='strong_wolfe')
    else:
        raise ValueError(f'Optimizer {opt_name} not supported')

"""
Helper function for parsing a mixed list of strings and numerical values. 

INPUT: 
- params_list: list of strings
OUTPUT: 
- params_dict: dictionary
"""
def parse_params_list(params_list): 
    # return an empty dictionary if there is no parameters specified
    if params_list is None: 
        return {}

    # parse parameter names and specified (if any) values
    params_dict = {}
    current_parameter = None
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    for token in params_list: 
        # attempt to extract a number from the token
        parsed_number = re.search(match_number, token)
        # if no match is found, then the token is a parameter name
        if parsed_number is None:
            params_dict[token] = None
            current_parameter = token
        # if the token indeed is a number (integer, decimal, or in scientific notation)
        else: 
            # append to the list of values associated with current parameter
            params_dict[current_parameter] = float(parsed_number.group())
    
    return params_dict

"""
Helper function for forming optimizer parameters. 

INPUT: 
- opt_params_list: list of strings
OUTPUT: 
- opt_params: dictionary
"""
def get_opt_params(opt_params_list): 
    return parse_params_list(opt_params_list)

def train(model,
          proj_name,
          pde_name,
          pde_params,
          loss_name,
          opt_name,
          opt_params_list,
          n_x,
          n_t,
          batch_size,
          num_epochs,
          device):
    model.apply(init_weights)

    x_range, t_range, loss_func, pde_coefs = get_pde(pde_name, pde_params, loss_name)
    opt_params = get_opt_params(opt_params_list)
    opt = get_opt(opt_name, opt_params, model.parameters())

    # TODO: Account for different values of batch_size

    x, t = get_data(x_range, t_range, n_x, n_t, random=False, device=device)
    wandb.log({'x': x, 't': t}) # Log training set

    # Store initial weights and loss
    save_folder = os.path.join(os.path.abspath(f'./{proj_name}_temp'), wandb.run.id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    wandb.log({'loss': loss_func(x, t, predict(x, t, model))})

    weights_hist = []
    weights_hist.append([p.detach().cpu().numpy() for p in model.parameters()])
    
    for i in range(num_epochs):
        model.train()
        def closure():
            opt.zero_grad()
            outputs = predict(x, t, model)
            loss = loss_func(x, t, outputs)
            loss.backward()

            return loss
        opt.step(closure)

        # record model parameters and loss
        model.eval()
        weights_hist.append([p.detach().cpu().numpy()
                            for p in model.parameters()])
        wandb.log({'loss': loss_func(x, t, predict(x, t, model))})
    
    # evaluate errors
    with torch.no_grad():
        predictions = predict(x, t, model)[0].cpu().detach().numpy()
    targets = get_ref_solutions(pde_name, pde_coefs, x, t)
    train_l1re = l1_relative_error(predictions, targets)
    train_l2re = l2_relative_error(predictions, targets)

    x_test, t_test = get_data(x_range, t_range, n_x, n_t, random=True, device=device)
    with torch.no_grad():
        predictions = predict(x_test, t_test, model)[0].cpu().detach().numpy()
    targets = get_ref_solutions(pde_name, pde_coefs, x_test, t_test)
    test_l1re = l1_relative_error(predictions, targets)
    test_l2re = l2_relative_error(predictions, targets)

    wandb.log({'train/l1re': train_l1re,
                'train/l2re': train_l2re,
                'test/l1re': test_l1re, 
                'test/l2re': test_l2re})
    
    # Save weights history as bz2 file
    filename = os.path.join(save_folder, 'weights_history.bz2')
    serialized_data = pickle.dumps(weights_hist)
    with bz2.open(filename, 'wb') as f:
        f.write(serialized_data)

    # Save file to wandb
    # wandb.save(os.path.abspath(filename))