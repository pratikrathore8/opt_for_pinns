import numpy as np
import torch
import torch.nn as nn

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
- x: tutple of (x_res, x_left, x_right, x_upper, x_lower)
- t: tutple of (t_res, t_left, t_right, t_upper, t_lower)
where: 
> res: numpy array / tensor of size (x_num * t_num) * 2; residual points -- all of the grid points
> b_left: numpy array / tensor of size (x_num) * 2; initial points (corresponding to initial time step)
> b_right: numpy array / tensor of size (x_num) * 2; terminal points (corresponding to terminal time step)
> b_upper: numpy array / tensor of size (t_num) * 2; upper boundary points
> b_lower: numpy array / tensor of size (t_num) * 2; lower boundary points
"""
def get_data(x_range, t_range, x_num, t_num, random=False, device='cpu'):
  if random: 
    x = np.concatenate((x_range[0], np.random.uniform(x_range[0], x_range[1], x_num-2), x_range[1]))
    t = np.concatenate((t_range[0], np.random.uniform(t_range[0], t_range[1], t_num-2), t_range[1]))
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
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.0)

"""
Helper function for making predictions with PINN. 

INPUT: 
- x: tutple of (x_res, x_left, x_right, x_upper, x_lower)
- t: tutple of (t_res, t_left, t_right, t_upper, t_lower)
- model: PINN model
OUTPUT: 
- preds: tutple of (pred_res, pred_left, pred_right, pred_upper, pred_lower)
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