import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from .opts.adam_lbfgs import Adam_LBFGS
from .opts.adam_lbfgs_nncg import Adam_LBFGS_NNCG
from .opts.adam_lbfgs_gd import Adam_LBFGS_GD
import random
import re
import wandb

LOG_FREQ = 20 # Hard-coded for now -- this is done to match the max_iter of the LBFGS optimizer + save time

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
            x_res, x_left, x_upper, x_lower = x
            t_res, t_left, t_upper, t_lower = t
            outputs_res, outputs_left, outputs_upper, outputs_lower = pred

            u_x = torch.autograd.grad(outputs_res, x_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]
            u_t = torch.autograd.grad(outputs_res, t_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]

            loss_res = loss_type["res"](u_t + pde_coefs["beta"] * u_x, torch.zeros_like(u_t))
            loss_bc = loss_type["bc"](outputs_upper - outputs_lower, torch.zeros_like(outputs_upper))
            loss_ic = loss_type["ic"](outputs_left[:,0], torch.sin(x_left[:,0]))

            # loss = loss_res + loss_bc + loss_ic

            return loss_res, loss_bc, loss_ic

    elif pde_name == "reaction": 
        if "rho" not in pde_coefs.keys(): 
            raise KeyError("rho is not specified for reaction PDE.")

        x_range = [0, 2 * np.pi]
        t_range = [0, 1]

        def loss_func(x, t, pred): 
            x_res, x_left, x_upper, x_lower = x
            t_res, t_left, t_upper, t_lower = t
            outputs_res, outputs_left, outputs_upper, outputs_lower = pred

            u_t = torch.autograd.grad(outputs_res, t_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]

            loss_res = loss_type["res"](u_t - pde_coefs["rho"] * outputs_res * (1 - outputs_res), torch.zeros_like(u_t))
            loss_bc = loss_type["bc"](outputs_upper - outputs_lower, torch.zeros_like(outputs_upper))
            loss_ic = loss_type["ic"](outputs_left[:,0], torch.exp(-(1/2) * torch.square((x_left[:,0] - np.pi) / (np.pi / 4))))

            # loss = loss_res + loss_bc + loss_ic

            return loss_res, loss_bc, loss_ic

    elif pde_name == "wave":
        if "beta" not in pde_coefs.keys():
            raise KeyError("beta is not specified for wave PDE.")

        x_range = [0, 1]
        t_range = [0, 1]

        def loss_func(x, t, pred):
            x_res, x_left, x_upper, x_lower = x
            t_res, t_left, t_upper, t_lower = t
            outputs_res, outputs_left, outputs_upper, outputs_lower = pred

            u_x = torch.autograd.grad(outputs_res, x_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]
            u_t = torch.autograd.grad(outputs_res, t_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]
            u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(outputs_res), retain_graph=True, create_graph=True)[0]

            loss_res = loss_type["res"](u_tt - 4 * u_xx, torch.zeros_like(u_tt))
            loss_bc = loss_type["bc"](outputs_upper, torch.zeros_like(outputs_upper)) + loss_type["bc"](outputs_lower, torch.zeros_like(outputs_lower))

            ui_t = torch.autograd.grad(outputs_left, t_left, grad_outputs=torch.ones_like(outputs_left), retain_graph=True, create_graph=True)[0]

            loss_ic_1 = loss_type["ic"](outputs_left[:,0], torch.sin(np.pi * x_left[:,0]) + 0.5 * torch.sin(pde_coefs["beta"] * np.pi * x_left[:,0]))
            loss_ic_2 = loss_type["ic"](ui_t, torch.zeros_like(ui_t))

            loss_ic = loss_ic_1 + loss_ic_2

            return loss_res, loss_bc, loss_ic

    else: 
        raise RuntimeError("{} is not a valid PDE name.".format(pde_name))

    return x_range, t_range, loss_func, pde_coefs

"""
Helper function for computing reference solution to the given PDE at given points. 

INPUT: 
- pde_name: string; name of the PDE problem
- pde_coefs: dictionary containing coefficients of the PDE
- x: tuple of (x_res, x_left, x_upper, x_lower)
- t: tuple of (t_res, t_left, t_upper, t_lower)
- data_params: dictionary containing parameters used to generate the data
OUTPUT: 
- sol: 
"""
def get_ref_solutions(pde_name, pde_coefs, x, t, data_params): 
    if pde_name == "convection": 
        sol = np.vstack([np.sin(x[i].cpu().detach().numpy() - pde_coefs["beta"] * t[i].cpu().detach().numpy()) for i in range(len(x))])
        
    elif pde_name == "reaction": 
        def compute_sol(x, t): 
            initial_func_term = np.exp(-(1/2) * np.square((x - np.pi) / (np.pi / 4)))
            exp_term = np.exp(pde_coefs['rho'] * t)
            return initial_func_term * exp_term / (initial_func_term * exp_term + 1 - initial_func_term)
        
        sol = np.vstack([compute_sol(x[i].cpu().detach().numpy(), t[i].cpu().detach().numpy()) for i in range(len(x))])

    elif pde_name == "wave":
        def compute_sol(x, t):
            return np.sin(np.pi * x) * np.cos(2 * np.pi * t) \
                + 0.5 * np.sin(pde_coefs["beta"] * np.pi * x) * np.cos(2 * pde_coefs["beta"] * np.pi * t)

        sol = np.vstack([compute_sol(x[i].cpu().detach().numpy(), t[i].cpu().detach().numpy()) for i in range(len(x))])

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

INPUT: 
- x_range: list of size 2; lower and upper bounds of spatial variable x
- t_range: list of size 2; lower and upper bounds of temporal variable t
- x_num: positive integer; number of x points
- t_num: positive integer; number of t points
- random: boolean; indication whether to (uniformly) randomly from the grid
- num_res_samples: positive integer; number of random samples to draw for residual points
- device: string; the device that the samples will be stored at
OUTPUT: 
- x: tuple of (x_res, x_left, x_upper, x_lower)
- t: tuple of (t_res, t_left, t_upper, t_lower)
- data_params: dictionary containing parameters used to generate the data 
               including x_range, t_range, x_num, t_num, grid_multiplier, and res_idx
where: 
> res: numpy array / tensor of size (x_num-2)(t_num-1) * 2 or num_res_samples * 2; residual points (interior grid or random samples from it)
> b_left: numpy array / tensor of size (x_num) * 2; initial points (corresponding to initial time step)
> b_upper: numpy array / tensor of size (t_num) * 2; upper boundary points
> b_lower: numpy array / tensor of size (t_num) * 2; lower boundary points
> res_idx: numpy array of length (x_num-2)(t_num-1) or num_res_samples; corresponding indices of the sampled residual points from the interior grid
"""
def get_data(x_range, t_range, x_num, t_num, random=False, num_res_samples=1e4, device='cpu'):
    # generate initial and boundary points
    x = np.linspace(x_range[0], x_range[1], x_num).reshape(-1, 1)
    t = np.linspace(t_range[0], t_range[1], t_num).reshape(-1, 1)
    # initial time
    x_left = x.copy()
    t_left = t_range[0] * np.ones([x_num,1])
    # lower boundary
    x_lower = x_range[0] * np.ones([t_num,1])
    t_lower = t.copy()
    # upper boundary
    x_upper = x_range[1] * np.ones([t_num,1])
    t_upper = t.copy()
    # residual points
    x_mesh, t_mesh = np.meshgrid(x[1:-1], t[1:])
    data_params = {
        "x_range": x_range, 
        "t_range": t_range, 
        "x_num": x_num, 
        "t_num": t_num
    }
    if random: 
        mesh = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))
        idx = np.random.choice(mesh.shape[0], num_res_samples, replace=False)
        x_res = mesh[idx, 0:1]
        t_res = mesh[idx, 1:2]
        data_params["res_idx"] = idx
    else: 
        x_res = x_mesh.reshape(-1,1)
        t_res = t_mesh.reshape(-1,1)
        data_params["res_idx"] = np.arange((x_num - 2) * (t_num - 1))

    # move data to target device
    if device != 'cpu': 
        x_left = torch.tensor(x_left, dtype=torch.float32, requires_grad=True).to(device)
        t_left = torch.tensor(t_left, dtype=torch.float32, requires_grad=True).to(device)
        x_upper = torch.tensor(x_upper, dtype=torch.float32, requires_grad=True).to(device)
        t_upper = torch.tensor(t_upper, dtype=torch.float32, requires_grad=True).to(device)
        x_lower = torch.tensor(x_lower, dtype=torch.float32, requires_grad=True).to(device)
        t_lower = torch.tensor(t_lower, dtype=torch.float32, requires_grad=True).to(device)
        x_res = torch.tensor(x_res, dtype=torch.float32, requires_grad=True).to(device)
        t_res = torch.tensor(t_res, dtype=torch.float32, requires_grad=True).to(device)

    # form tuples
    x = (x_res, x_left, x_upper, x_lower)
    t = (t_res, t_left, t_upper, t_lower)

    return x, t, data_params

"""
Helper function for initializing neural net parameters. 
"""
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

"""
Helper function for making predictions with PINN. 

INPUT: 
- x: tuple of (x_res, x_left, x_upper, x_lower)
- t: tuple of (t_res, t_left, t_upper, t_lower)
- model: PINN model
OUTPUT: 
- preds: tuple of (pred_res, pred_left, pred_upper, pred_lower)
where: 
> pred_res: predictions on residual points
> pred_left: predictions on initial points
> pred_upper: predictions on upper boundary points
> pred_lower: predictions on lower boundary points
"""
def predict(x, t, model): 
    x_res, x_left, x_upper, x_lower = x
    t_res, t_left, t_upper, t_lower = t
    
    pred_res = model(x_res, t_res)
    pred_left = model(x_left, t_left)
    pred_upper = model(x_upper, t_upper)
    pred_lower = model(x_lower, t_lower)
    
    preds = (pred_res, pred_left, pred_upper, pred_lower) 
    
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
    elif opt_name == 'adam_lbfgs':
        if "switch_epochs" not in opt_params:
            raise KeyError("switch_epochs is not specified for Adam_LBFGS optimizer.")
        switch_epochs = opt_params["switch_epochs"]

        # Ensure switch_epochs is a list of integers
        if not isinstance(switch_epochs, list):
            switch_epochs = [switch_epochs]
        switch_epochs = [int(epoch) for epoch in switch_epochs]

        # Get parameters for Adam and LBFGS, remove the prefix "adam_" and "lbfgs_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        
        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])

        return Adam_LBFGS(model_params, switch_epochs, adam_params, lbfgs_params)
    elif opt_name == 'adam_lbfgs_nncg':
        if "switch_epoch_lbfgs" not in opt_params:
            raise KeyError("switch_epoch_lbfgs is not specified for Adam_LBFGS_NNCG optimizer.")
        if "switch_epoch_nncg" not in opt_params:
            raise KeyError("switch_epoch_nncg is not specified for Adam_LBFGS_NNCG optimizer.")
        if "precond_update_freq" not in opt_params:
            raise KeyError("precond_update_freq is not specified for Adam_LBFGS_NNCG optimizer.")
        switch_epoch_lbfgs = int(opt_params["switch_epoch_lbfgs"])
        switch_epoch_nncg = int(opt_params["switch_epoch_nncg"])
        precond_update_freq = int(opt_params["precond_update_freq"])

        # Get parameters for Adam, LBFGS, and NNCG, remove the prefix "adam_", "lbfgs_", and "nncg_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        nncg_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("nncg_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        nncg_params["line_search_fn"] = "armijo"

        nncg_params["verbose"] = True

        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])
        if "rank" in nncg_params:
            nncg_params["rank"] = int(nncg_params["rank"])

        return Adam_LBFGS_NNCG(model_params, switch_epoch_lbfgs, switch_epoch_nncg, precond_update_freq, adam_params, lbfgs_params, nncg_params)
    elif opt_name == 'adam_lbfgs_gd':
        if "switch_epoch_lbfgs" not in opt_params:
            raise KeyError("switch_epoch_lbfgs is not specified for Adam_LBFGS_GD optimizer.")
        if "switch_epoch_gd" not in opt_params:
            raise KeyError("switch_epoch_gd is not specified for Adam_LBFGS_GD optimizer.")
        switch_epoch_lbfgs = int(opt_params["switch_epoch_lbfgs"])
        switch_epoch_gd = int(opt_params["switch_epoch_gd"])

        # Get parameters for Adam, LBFGS, and GD, remove the prefix "adam_", "lbfgs_", and "gd_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        gd_params = {k[3:]: v for k, v in opt_params.items() if k.startswith("gd_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        gd_params["line_search_fn"] = "armijo"

        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])

        return Adam_LBFGS_GD(model_params, switch_epoch_lbfgs, switch_epoch_gd, adam_params, lbfgs_params, gd_params)
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
            # if the current parameter is not specified yet, then the number is the value of the current parameter
            # otherwise, the number is appended to the list of values associated with current parameter
            if params_dict[current_parameter] is not None:
                if not isinstance(params_dict[current_parameter], list):
                    params_dict[current_parameter] = [params_dict[current_parameter]]
                params_dict[current_parameter].append(float(parsed_number.group()))
            else:
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

"""
Helper function for getting the list of logging times.

INPUT:
- opt: optimizer to be used
- log_freq: integer; logging frequency
OUTPUT:
- log_times: list of positive integers; logging times
"""
# TODO: Make this robust to when log_freq does not match the max_iter of the LBFGS optimizer
def get_log_times(opt, log_freq, num_epochs):
    log_times = []
    if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)):
        # Get times up to opt.switch_epoch1, starting at log_freq (Adam)
        log_times = list(range(log_freq - 1, opt.switch_epoch1, log_freq))
        # Get times up to opt.switch_epoch2 (possibly including opt.switch_epoch2), starting at opt.switch_epoch1 (L-BFGS)
        log_times += list(range(opt.switch_epoch1, opt.switch_epoch2, 1))
        # Get times up to num_epochs (possibly including num_epochs), starting at opt.switch_epoch2 (NNCG/GD)
        log_times += list(range(opt.switch_epoch2 + log_freq - 1, num_epochs, log_freq))
    elif isinstance(opt, Adam_LBFGS):
        # Get times up to opt.switch_epochs[0], starting at log_freq (Adam)
        log_times = list(range(log_freq - 1, opt.switch_epochs[0], log_freq))
        # Get times up to num_epochs (possibly including num_epochs), starting at opt.switch_epochs[0] (L-BFGS)
        log_times += list(range(opt.switch_epochs[0], num_epochs, 1))
    elif isinstance(opt, Adam):
        # Get times up to num_epochs (possibly including num_epochs), starting at log_freq (Adam)
        log_times = list(range(log_freq - 1, num_epochs, log_freq))
    elif isinstance(opt, LBFGS):
        # Get all times up to num_epochs (possibly including num_epochs), starting at 1 (L-BFGS)
        log_times = list(range(0, num_epochs, 1))

    return log_times

def train(model,
          proj_name,
          pde_name,
          pde_params,
          loss_name,
          opt_name,
          opt_params_list,
          n_x,
          n_t,
          n_res,
          batch_size,
          num_epochs,
          device):
    model.apply(init_weights)

    x_range, t_range, loss_func, pde_coefs = get_pde(pde_name, pde_params, loss_name)
    opt_params = get_opt_params(opt_params_list)
    opt = get_opt(opt_name, opt_params, model.parameters())

    logging_times = get_log_times(opt, LOG_FREQ, num_epochs)

    # TODO: Account for different values of batch_size

    x, t, data_params = get_data(x_range, t_range, n_x, n_t, random=True, num_res_samples=n_res, device=device)
    wandb.log({'x': x, 't': t}) # Log training set

    loss_res, loss_bc, loss_ic = loss_func(x, t, predict(x, t, model))
    loss = loss_res + loss_bc + loss_ic
    wandb.log({'loss': loss.item(),
            'loss_res': loss_res.item(),
            'loss_bc': loss_bc.item(),
            'loss_ic': loss_ic.item()})
    
    for i in range(num_epochs):
        model.train()

        # Update the preconditioner for NysNewtonCG
        if isinstance(opt, Adam_LBFGS_NNCG) and i >= opt.switch_epoch2 and i % opt.precond_update_freq == 0:
            opt.zero_grad()
            outputs = predict(x, t, model)
            loss_res, loss_bc, loss_ic = loss_func(x, t, outputs)
            loss = loss_res + loss_bc + loss_ic
            grad_tuple = torch.autograd.grad(
                loss, model.parameters(), create_graph=True)
            opt.nncg.update_preconditioner(grad_tuple)

        # Separate closure is needed for NysNewtonCG/GD
        if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)) and i >= opt.switch_epoch2:
            def closure():
                opt.zero_grad()
                outputs = predict(x, t, model)
                loss_res, loss_bc, loss_ic = loss_func(x, t, outputs)
                loss = loss_res + loss_bc + loss_ic
                grad_tuple = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                return loss, grad_tuple
        else:
            def closure():
                opt.zero_grad()
                outputs = predict(x, t, model)
                loss_res, loss_bc, loss_ic = loss_func(x, t, outputs)
                loss = loss_res + loss_bc + loss_ic
                loss.backward()
                return loss

        if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)) and i >= opt.switch_epoch2:
            grad = opt.step(closure)
        else:
            opt.step(closure)

        # record model parameters and loss
        model.eval()
        if i in logging_times:
            loss_res, loss_bc, loss_ic = loss_func(x, t, predict(x, t, model))
            loss = loss_res + loss_bc + loss_ic

            # Compute the gradient norm of the full objective function
            # NOTE: This will not work if we do minibatching
            if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)) and i >= opt.switch_epoch2:
                grad_norm = torch.norm(grad).item()
            else:
                grad_norm = 0
                for p in model.parameters():
                    grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5

            if isinstance(opt, Adam_LBFGS_NNCG) and i >= opt.switch_epoch2:
                wandb.log({'step_size': opt.nncg.state_dict()['state'][0]['t']},
                            commit=False)
            elif isinstance(opt, Adam_LBFGS_GD) and i >= opt.switch_epoch2:
                wandb.log({'step_size': opt.gd.state_dict()['state'][0]['t']},
                            commit=False)

            wandb.log({'loss': loss.item(),
                'loss_res': loss_res.item(),
                'loss_bc': loss_bc.item(),
                'loss_ic': loss_ic.item(),
                'grad_norm': grad_norm})
    
    # evaluate errors
    with torch.no_grad():
        predictions = torch.vstack(predict(x, t, model)).cpu().detach().numpy()
    targets = get_ref_solutions(pde_name, pde_coefs, x, t, data_params)
    train_l1re = l1_relative_error(predictions, targets)
    train_l2re = l2_relative_error(predictions, targets)

    # coarse grid for testing
    n_x_test = int((n_x - 1) / 2) + 1
    n_t_test = n_t
    x_test, t_test, data_params_test = get_data(x_range, t_range, n_x_test, n_t_test, random=False, device=device)
    with torch.no_grad():
        predictions = torch.vstack(predict(x_test, t_test, model)).cpu().detach().numpy()
    targets = get_ref_solutions(pde_name, pde_coefs, x_test, t_test, data_params_test)
    test_l1re = l1_relative_error(predictions, targets)
    test_l2re = l2_relative_error(predictions, targets)

    wandb.log({'train/l1re': train_l1re, 
                'train/l2re': train_l2re, 
                'test/l1re': test_l1re, 
                'test/l2re': test_l2re})