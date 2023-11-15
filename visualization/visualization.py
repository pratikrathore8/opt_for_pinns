import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from pyhessian import hessian

"""
Helper function for projecting trajectories onto their first two principle components. 

INPUT: 
- weights_hist: list of list of PyTorch Tensors; model parameters at the beginning of each epoch
- final_weights: list of PyTorch Tensors; final model parameters of the trained model
- device: string; the device that weights are stored at
OUTPUT: 
- pc_directions: numpy array of size 2 * num_of_model_params; first two PCs of the trajectories
- variance_ratios: numpy array of size 2; percentage of the variance explained by each PC
- trajectory_coords: numpy array of size 2 * num_epochs; 
                     coordinates (coefficients) of the projected trajectory in each PC direction
"""
def project_trajectories(weights_hist, final_weights, device='cuda'):
  # form trajectory matrix (difference of the model weights)
  traj_matrix = []
  for weights in weights_hist: 
    # compute difference with the final weights
    diff = [w - w_f for (w_f, w) in zip(final_weights, weights)]
    flattened_diff = torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.Tensor(w).type(type(w)).cuda(device=device) for w in diff])
    traj_matrix.append(flattened_diff.cpu().numpy())
  
  # run PCA to find the first two principal components
  n_components = 2
  pca = PCA(n_components=n_components)
  pca.fit(np.array(traj_matrix))

  # project trajectories onto the found directions (using cosine)
  pc_norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=pca.components_)
  proj_coords = np.zeros([n_components, len(weights_hist)])
  for idx, trajectory in enumerate(traj_matrix): 
    proj_coords[:,idx] = pca.components_ @ trajectory / pc_norms

  return pca.components_, pca.explained_variance_ratio_, proj_coords


"""
Helper function for generating the grid of losses which will be used to make the plot. 

INPUT: 
- directions: 2 * num_of_model_params; two directions to be used to generate the loss grid
- final_weights: list of PyTorch Tensors; final model parameters of the trained model
- data: touple of spatial and temporal variables; data to be used to evaluate loss (needs to be stored in the same device as the model)
- pred_func: prediction function that takes (x,t,model) and outputs predictions
- loss_func: loss function that takes (x,t,pred) and computes the total loss
- model: PINN model
- device: string; device name that weights, data, and model are stored at
- num_samples_x: positive integer; number of grid points in the x direction
- num_samples_y: positive integer; number of grid points in the y direction
- sample_range_x: list of size two; lower and upper bounds of the grid in the x direction
- sample_range_y: list of size two; lower and upper bounds of the grid in the y direction
- log_scale: boolean; indication whether log sccale should be used (for losses; base 10)
OUTPUT: 
- loss_grid: numpy array of size (num_samples_y * 1) * (num_samples_x + 1); array of losses at each grid point
- coord_grid: tuple of numpy arrays of size (num_samples_y + 1) * (num_samples_x * 1); array of coordinates at each grid point
"""
def generate_loss_grid(directions, final_weights, data, pred_func, loss_func, model, device='cuda', num_samples_x=100, num_samples_y=100, sample_range_x=[-1,1], sample_range_y=[-1,1], log_scale=True):
  # unpack inputs
  x, t = data

  # generate grid
  xcoords = np.linspace(sample_range_x[0], sample_range_x[1], num=num_samples_x+1)
  ycoords = np.linspace(sample_range_y[0], sample_range_y[1], num=num_samples_y+1)
  xcoord_mesh, ycoord_mesh = np.meshgrid(xcoords, ycoords)
  coordinates = np.c_[xcoord_mesh.ravel(), ycoord_mesh.ravel()]

  # compute loss at each point on the grid
  loss_grid = []
  for coord_change in coordinates: 
    # update model parameters
    change = coord_change.dot(directions)
    offset = 0
    for (p, w) in zip(model.parameters(), final_weights):
      numel = w.numel()
      step = change[offset:offset+numel]
      p.data = w + torch.Tensor(step).type(type(w)).view(w.size()).cuda(device=device)
      offset += numel

    # evaluate loss
    outputs = pred_func(x, t, model)
    loss = loss_func(x, t, outputs).item()

    # add loss to the grid
    loss_grid.append(loss)

  loss_grid = np.array(loss_grid).reshape(len(ycoords), len(xcoords))
  # convert to log scale
  if log_scale: 
    loss_grid = np.log10(loss_grid)

  return loss_grid, (xcoord_mesh, ycoord_mesh)

"""
Helper function for making trajectory plot and animation. 

INPUT: 
- loss_grid: numpy array of size (num_samples_y + 1) * (num_samples_x * 1); array of losses at each grid point
- coord_grid: tuple of numpy arrays of size (num_samples_y + 1) * (num_samples_x * 1); array of coordinates at each grid point
- trajectory_coords: numpy array of size 2 * num_epochs; coordinates of the projected trajectory in each PC direction
- variance_ratios: numpy array of size 2; percentage of the variance explained by each PC direction
- countour_range: list of size two; lower and upper bounds of the countour levels
- num_contours: positive integer; number of levels to be used for the plot
- save_file: string; name of the file
- animate: boolean; indication whether to animate the plot
OUTPUT: 
- None
"""
def plot_trajectories(loss_grid, coord_grid, trajectory_coords, variance_ratios, countour_range=None, num_contours=100, save_file="trajectories", animate=False, animation_format='gif'): 
  # generate contour levels
  if countour_range is None: 
    countour_range = [np.floor(np.min(loss_grid)), np.ceil(np.max(loss_grid))]
  countour_levels = np.linspace(countour_range[0], countour_range[1], num_contours)
  
  # make trajectory plot
  fig = plt.figure()
  # plot loss landscape
  contour_plot = plt.contour(coord_grid[0], coord_grid[1], loss_grid, cmap='summer', levels=countour_levels)
  plt.clabel(contour_plot, inline=1, fontsize=8)
  # plot trajectories
  plt.plot(trajectory_coords[0], trajectory_coords[1], marker='.', color='firebrick', markersize=10)
  plt.xlabel('1st PC: %.2f %%' % (variance_ratios[0]*100), fontsize='large')
  plt.ylabel('2nd PC: %.2f %%' % (variance_ratios[1]*100), fontsize='large')
  # save file
  if save_file is not None: 
    fig.savefig(save_file + ".pdf", dpi=300, bbox_inches='tight', format='pdf')

  # make animation
  if animate: 
    num_trajectories = trajectory_coords.shape[1]
    fig = plt.figure()
    # plot still components
    contour_plot = plt.contour(coord_grid[0], coord_grid[1], loss_grid, cmap='summer', levels=countour_levels)
    plt.plot(trajectory_coords[0], trajectory_coords[1], marker='None', color='firebrick', alpha=0.35, linewidth=1)
    plt.xlabel('1st PC: %.2f %%' % (variance_ratios[0]*100), fontsize='large')
    plt.ylabel('2nd PC: %.2f %%' % (variance_ratios[1]*100), fontsize='large')
    # plot animated components
    graph, = plt.plot([], [], marker='.', color='firebrick', markersize=8)
    def animate(i):
      graph.set_data(trajectory_coords[0,:i+1], trajectory_coords[1,:i+1])
      i += 1
      return graph
    animation = FuncAnimation(fig, animate, frames=num_trajectories, interval=50)
    if save_file is not None: 
      animation.save(save_file + "." + animation_format, dpi=120, writer="pillow")


"""
Helper function for obtaining the top two eigenvectors of the Hessian. 

INPUT: 
- data: touple of spatial and temporal variables; data to be used to evaluate loss (needs to be stored in the same device as the model)
- pred_func: prediction function that takes (x,t,model) and outputs predictions
- loss_func: loss function that takes (x,t,pred) and computes the total loss
- model: PINN model
- device: string; device name that data and model are stored at
OUTPUT: 
- directions: list of size 2; top two eigenvectors of the Hessian
"""
def compute_hessian_directions(data, pred_func, loss_func, model, device='cuda'): 
  use_cuda = False if device=='cpu' else True
  hessian_comp = hessian(model, pred_func, loss_func, data, cuda=use_cuda, device=device)
  _, eig_vecs = hessian_comp.eigenvalues(maxIter=100, tol=1e-3, top_n=2)

  # flatten the eigenvectors
  directions = []
  directions.append(torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.Tensor(w).type(type(w)).cuda(device=device) for w in eig_vecs[0]]).cpu().numpy())
  directions.append(torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.Tensor(w).type(type(w)).cuda(device=device) for w in eig_vecs[1]]).cpu().numpy())

  return directions


"""
Helper function for making loss landscape plot. 

INPUT: 
- loss_grid: numpy array of size (num_samples_x + 1) * (num_samples_y * 1); array of losses at each grid point
- coord_grid: tuple of numpy arrays of size (num_samples_y + 1) * (num_samples_x * 1); array of coordinates at each grid point
- countour_range: list of size two; lower and upper bounds of the countour levels
- num_contours: positive integer; number of levels to be used for the plot
- save_file: string; name of the file
OUTPUT: 
- None
"""
def plot_landscape(loss_grid, coord_grid, countour_range=None, num_contours=100, save_file="landscape"): 
  # generate contour levels
  if countour_range is None: 
    countour_range = [np.floor(np.min(loss_grid)), np.ceil(np.max(loss_grid))]
  countour_levels = np.linspace(countour_range[0], countour_range[1], num_contours)
  
  # make the plot
  fig = plt.figure()
  # plot loss landscape
  contour_plot = plt.contour(coord_grid[0], coord_grid[1], loss_grid, cmap='summer', levels=countour_levels)
  plt.clabel(contour_plot, inline=1, fontsize=8)
  # save file
  if save_file is not None: 
    fig.savefig(save_file + ".pdf", dpi=300, bbox_inches='tight', format='pdf')