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

    elif pde_name == "reaction_diffusion": 
        if not {"nu", "rho"} <= pde_coefs.keys(): 
            raise KeyError("nu or rho is not specified for reaction diffusion PDE.")

        x_range = [0, 2 * np.pi]
        t_range = [0, 1]

        def loss_func(x, t, pred): 
            x_res, x_left, x_right, x_upper, x_lower = x
            t_res, t_left, t_right, t_upper, t_lower = t
            outputs_res, outputs_left, outputs_right, outputs_upper, outputs_lower = pred

            u_x = torch.autograd.grad(outputs_res, x_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]
            u_t = torch.autograd.grad(outputs_res, t_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]

            loss_res = loss_type["res"](u_t - pde_coefs["nu"] * u_xx - pde_coefs["rho"] * outputs_res * (1 - outputs_res), torch.zeros_like(u_t))
            loss_bc = loss_type["bc"](outputs_upper - outputs_lower, torch.zeros_like(outputs_upper))
            loss_ic = loss_type["ic"](outputs_left[:,0], torch.exp(-(1/2) * torch.square((x_left[:,0] - np.pi) / (np.pi / 4))))

            loss = loss_res + loss_bc + loss_ic

        return loss

    else: 
        raise RuntimeError("{} is not a valid PDE name.".format(pde_name))

    return x_range, t_range, loss_func, pde_coefs

"""
Helper function for computing reference solution to the given PDE at given points. 

INPUT: 
- pde_name: string; name of the PDE problem
- pde_coefs: dictionary containing coefficients of the PDE
- x: tuple of (x_res, x_left, x_right, x_upper, x_lower)
- t: tuple of (t_res, t_left, t_right, t_upper, t_lower)
- data_params: dictionary containing parameters used to generate the data
OUTPUT: 
- sol: 
"""
def get_ref_solutions(pde_name, pde_coefs, x, t, data_params): 
    if pde_name == "convection": 
        sol = np.vstack([np.sin(x[i].cpu().detach().numpy() - pde_coefs["beta"] * t[i].cpu().detach().numpy()) for i in range(len(x))])
    elif pde_name == "reaction_diffusion": 
        # unpack data-generation parameters
        x_range = data_params["x_range"]
        t_range = data_params["t_range"]
        x_num = data_params["x_num"]
        t_num = data_params["t_num"]
        grid_multiplier = data_params["grid_multiplier"]
        res_idx = data_params["res_idx"]
        # generate grid
        x = np.linspace(x_range[0], x_range[1], x_num * grid_multiplier).reshape(-1, 1)
        t = np.linspace(t_range[0], t_range[1], t_num * grid_multiplier).reshape(-1, 1)
        x_mesh, t_mesh = np.meshgrid(x, t)
        # compute initial solution
        u0 = np.exp(-(1/2) * np.square((x - np.pi) / (np.pi / 4)))
        u = np.zeros((x_num * grid_multiplier, t_num * grid_multiplier))
        u[:,0] = u0

        IKX_pos = 1j * np.arange(0, (x_num * grid_multiplier) / 2 + 1, 1)
        IKX_neg = 1j * np.arange(-(x_num * grid_multiplier) / 2 + 1, 0, 1)
        IKX = np.concatenate((IKX_pos, IKX_neg))
        IKX2 = IKX * IKX
        # perform time-marching
        t_step_size = (t_range[1] - t_range[0]) / (t_num - 1)
        u_t = u_0.copy()
        for i in range(t_num * grid_multiplier - 1): 
            # reaction component
            factor = u_t * np.exp(pde_coefs['rho'] * t_step_size)
            u_t = factor / (factor + (1 - u_t))
            # diffusion component
            factor = np.exp(pde_coefs['nu'] * IKX2 * t_step_size)
            u_hat = np.fft.fft(u_t) * factor
            u_t = np.real(np.fft.ifft(u_hat))
            u[:,i+1] = u_t


         (x_res, x_left, x_right, x_upper, x_lower)

        sol_left = u[:,0].reshape(-1,1)
        sol_right = u[:,-1].reshape(-1,1)
        sol_upper = u[-1,:].reshape(-1,1)
        sol_lower = u[0,:].reshape(-1,1)
        sol_res = u[1:-1, 1:-1].T.reshape(-1,1)[res_idx]

        sol = np.vstack([sol_res, sol_left, sol_right, sol_upper, sol_lower])

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
- grid_multiplier: positive integer; multiplicative factor that determines granularity of a finer grid 
                   compared to the full grid; random subsamples of the residual points are drawn from the finer grid
- device: string; the device that the samples will be stored at
OUTPUT: 
- x: tuple of (x_res, x_left, x_right, x_upper, x_lower)
- t: tuple of (t_res, t_left, t_right, t_upper, t_lower)
- data_params: dictionary containing parameters used to generate the data 
               including x_range, t_range, x_num, t_num, grid_multiplier, and res_idx
where: 
> res: numpy array / tensor of size (x_num * t_num) * 2; residual points -- all of the grid points
> b_left: numpy array / tensor of size (x_num) * 2; initial points (corresponding to initial time step)
> b_right: numpy array / tensor of size (x_num) * 2; terminal points (corresponding to terminal time step)
> b_upper: numpy array / tensor of size (t_num) * 2; upper boundary points
> b_lower: numpy array / tensor of size (t_num) * 2; lower boundary points
> res_idx: numpy array of length (x_num - 2)(t_num - 2); corresponding indices of the sampled residual points from the finer grid
"""
def get_data(x_range, t_range, x_num, t_num, random=False, grid_multiplier=100, device='cpu'):
  # generate initial and boundary points
  x = np.linspace(x_range[0], x_range[1], x_num).reshape(-1, 1)
  t = np.linspace(t_range[0], t_range[1], t_num).reshape(-1, 1)
  # initial time
  x_left = x.copy()
  t_left = t_range[0] * np.ones([x_num,1])
  # terminal time
  x_right = x.copy()
  t_right = t_range[1] * np.ones([x_num,1])
  # lower boundary
  x_lower = x_range[0] * np.ones([t_num,1])
  t_lower = t.copy()
  # upper boundary
  x_upper = x_range[1] * np.ones([t_num,1])
  t_upper = t.copy()

  # generate residual points
  data_params = {
    "x_range": x_range, 
    "t_range": t_range, 
    "x_num": x_num, 
    "t_num": t_num
  }
  if random: 
    # generate finer grid
    x = np.linspace(x_range[0], x_range[1], x_num * grid_multiplier).reshape(-1, 1)
    t = np.linspace(t_range[0], t_range[1], t_num * grid_multiplier).reshape(-1, 1)
    x_mesh, t_mesh = np.meshgrid(x[1:-1], t[1:-1])
    # sub-sample randomly from the new grid
    mesh = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))
    idx = np.random.choice(mesh.shape[0], (x_num - 2) * (t_num - 2), replace=False)
    x_res = mesh[idx, 0:1]
    t_res = mesh[idx, 1:2]
    # update parameters used for data generation
    data_params["grid_multiplier"] = grid_multiplier
    data_params["res_idx"] = idx
  else: 
    # form interior grid
    x_mesh, t_mesh = np.meshgrid(x[1:-1], t[1:-1])
    x_res = x_mesh.reshape(-1,1)
    t_res = t_mesh.reshape(-1,1)
    # update parameters used for data generation
    data_params["grid_multiplier"] = 1
    data_params["res_idx"] = np.arange((x_num - 2) * (t_num - 2))

  # move data to target device
  if device != 'cpu': 
    x_left = torch.tensor(x_left, dtype=torch.float32, requires_grad=True).to(device)
    t_left = torch.tensor(t_left, dtype=torch.float32, requires_grad=True).to(device)
    x_right = torch.tensor(x_right, dtype=torch.float32, requires_grad=True).to(device)
    t_right = torch.tensor(t_right, dtype=torch.float32, requires_grad=True).to(device)
    x_upper = torch.tensor(x_upper, dtype=torch.float32, requires_grad=True).to(device)
    t_upper = torch.tensor(t_upper, dtype=torch.float32, requires_grad=True).to(device)
    x_lower = torch.tensor(x_lower, dtype=torch.float32, requires_grad=True).to(device)
    t_lower = torch.tensor(t_lower, dtype=torch.float32, requires_grad=True).to(device)
    x_res = torch.tensor(x_res, dtype=torch.float32, requires_grad=True).to(device)
    t_res = torch.tensor(t_res, dtype=torch.float32, requires_grad=True).to(device)

  # form tuples
  x = (x_res, x_left, x_right, x_upper, x_lower)
  t = (t_res, t_left, t_right, t_upper, t_lower)

  return x, t, data_params

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

"""
Helper function for initializing the optimizer with specified parameters. 

INPUT: 
- opt_name: string; name of the optimizer
- opt_params: dictionary; arguments used to initialize the optimizer
- model_params: dictionary; contains Tensors of the model to be optimized
OUTPUT: 
- opt: torch.optim.Optimizer instance
"""
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

    x, t, data_params = get_data(x_range, t_range, n_x, n_t, random=False, device=device)
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
        predictions = torch.vstack(predict(x, t, model)).cpu().detach().numpy()
    targets = get_ref_solutions(pde_name, pde_coefs, x, t, data_params)
    train_l1re = l1_relative_error(predictions, targets)
    train_l2re = l2_relative_error(predictions, targets)

    x_test, t_test, data_params_test = get_data(x_range, t_range, n_x, n_t, random=True, device=device)
    with torch.no_grad():
        predictions = torch.vstack(predict(x_test, t_test, model)).cpu().detach().numpy()
    targets = get_ref_solutions(pde_name, pde_coefs, x_test, t_test, data_params_test)
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