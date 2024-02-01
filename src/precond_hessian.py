import torch

class precond_hessian():
  """
  Class for computing spectral density of (L-BFGS) pre-conditioned Hessian. 

  - model: instance of PINN
  - pred_func: prediction function
  - loss_func: loss function
  - optimizer: L-BFGS instance used to optimize the model
  - data: tuple of spatial and temporal inputs
  - loss_comp: string indicating which component of the loss to use (will use total loss for any value other than "res", "bc", or "ic")
  - device: string indicating which CUDA device to use (where both model and data reside)
  """
  def __init__(self, model, pred_func, loss_func, optimizer, data, loss_comp=None, device='cuda'):
    self.model = model.eval() 
    self.pred_func = pred_func
    self.loss_func = loss_func
    self.optimizer = optimizer
    self.x, self.t = data
    self.device = device

    # pre-compute results needed for matrix-vector-products
    # compute components of tilde_H_k
    history_size = len(optimizer.state_dict()["state"][0]['old_dirs'])
    y_hist = optimizer.state_dict()["state"][0]['old_dirs']
    s_hist = optimizer.state_dict()["state"][0]['old_stps']
    rho_hist = optimizer.state_dict()["state"][0]['ro']
    gamma = optimizer.state_dict()["state"][0]['H_diag']

    tilde_v_vecs = [None] * history_size
    tilde_s_vecs = [None] * history_size

    tilde_v_vecs[-1] = s_hist[-1]
    tilde_s_vecs[-1] = rho_hist[-1].sqrt() * s_hist[-1]

    for i in range(history_size-2, -1, -1):
      # compute tilde_s_i
      tilde_s_updates = [y.dot(s_hist[i]) * rho * tilde_v for rho, y, tilde_v in zip(rho_hist[i+1:], y_hist[i+1:], tilde_v_vecs[i+1:])]
      tilde_s_vecs[i] = rho_hist[i].sqrt() * (s_hist[i] - torch.stack(tilde_s_updates, 0).sum(0))
      # compute tilde_v_i
      tilde_v_terms = [rho * y.dot(s_hist[i]) * tilde_v for rho, y, tilde_v in zip(rho_hist[i+1:], y_hist[i+1:], tilde_v_vecs[i+1:])]
      tilde_v_vecs[i] = s_hist[i] - torch.stack(tilde_v_terms, 0).sum(0)

    self.param_length = optimizer._numel_cache
    self.history_size = history_size
    self.y_hist = y_hist
    self.rho_hist = rho_hist
    self.gamma = gamma
    self.tilde_v_vecs = tilde_v_vecs
    self.tilde_s_vecs = tilde_s_vecs

    # get model parameters and gradient
    self.model.zero_grad()
    outputs = self.pred_func(self.x, self.t, self.model)
    loss_res, loss_bc, loss_ic = self.loss_func(self.x, self.t, outputs)
    if loss_comp == "res": 
      loss = loss_res + torch.nn.MSELoss()(outputs[0] * 0, torch.zeros_like(outputs[0]))  # second term ensures all model parameters are in the graph
    elif loss_comp == "bc": 
      loss = loss_bc
    elif loss_comp == "ic": 
      loss = loss_ic
    else: 
      loss = loss_res + loss_bc + loss_ic

    grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

    self.params = [param for param in self.model.parameters() if param.requires_grad]
    self.gradsH = [gradient if gradient is not None else 0.0 for gradient in grad_tuple]

  """
  Function for computing matrix vector product of matrix (tilde_H_k.T Hessian tilde_H_k) and vector v. 

  INPUT: 
  - v: tensor of size (num of model params + history size of L-BFGS)
  OUTPUT: 
  - mv: tensor of size (num of model params + history size of L-BFGS)
  """
  def matrix_vector_product(self, v): 
    # step 1: compute product of tilde_H_k and v
    # compute v_prime
    v1 = v[:-self.history_size]
    v2 = v[-self.history_size:]
    v_prime = [rho * y.dot(v1) for rho, y in zip(self.rho_hist, self.y_hist)]
    v_prime = v1 - torch.stack([v_i * tilde_v for v_i, tilde_v in zip(v_prime, self.tilde_v_vecs)], 0).sum(0)
    v_prime = self.gamma.sqrt() * v_prime + torch.stack([v_i * tilde_s for v_i, tilde_s in zip(v2, self.tilde_s_vecs)], 0).sum(0)
    
    # step 2: compute Hv_prime using HVP
    # convert tensor to a list of tensors matching model parameters
    v_prime_list = []
    offset = 0
    for p in self.params: 
      numel = p.numel()
      v_prime_list.append(v_prime[offset:offset + numel].view_as(p))
      offset += numel
    hv_prime = torch.autograd.grad(self.gradsH, self.params, grad_outputs=v_prime_list, only_inputs=True, retain_graph=True)
    # flatten result
    views = []
    for p in hv_prime:
      views.append(p.view(-1))
    hv_prime = torch.cat(views, 0)
    
    # step 3: compute product of tilde_H_k^T and Hv_prime
    v_prime = [tilde_v.dot(hv_prime) for tilde_v in self.tilde_v_vecs]
    v_prime = torch.stack([v_i * rho * y for rho, y, v_i in zip(self.rho_hist, self.y_hist, v_prime)], 0).sum(0)
    v_prime = self.gamma.sqrt() * (hv_prime - v_prime)
    v_double_prime = torch.tensor([tilde_s.dot(hv_prime) for tilde_s in self.tilde_s_vecs], device=self.device)
    mv = torch.cat([v_prime, v_double_prime], 0)

    return mv

  """
  Function for performing spectral density computation. 

  INPUT: 
  - num_iter: number of iterations for Lanczos
  - num_run: number of runs
  OUTPUT: 
  - eigen_list_full: list eigenvalues for each run
  - weight_list_full: list of corresponding densities for each run
  """
  def density(self, num_iter=100, num_run=1):
    eigen_list_full = []
    weight_list_full = []

    for k in range(num_run):
      # generate Rademacher random vector
      v = (2 * torch.randint(high=2, size=(self.param_length+self.history_size,), device=self.device) - 1).float()
      v = self._normalization(v)

      # Lanczos initlization
      v_list = [v]
      w_list = []
      alpha_list = []
      beta_list = []

      # run Lanczos
      for i in range(num_iter):
        self.model.zero_grad()
        w_prime = torch.zeros(self.param_length+self.history_size).to(self.device)
        if i == 0:
          w_prime = self.matrix_vector_product(v)
          alpha = w_prime.dot(v)
          alpha_list.append(alpha.cpu().item())
          w = w_prime - alpha * v
          w_list.append(w)
        else:
          beta = torch.sqrt(w.dot(w))
          beta_list.append(beta.cpu().item())
          if beta_list[-1] != 0.:
            v = self._orthonormalization(w, v_list)
            v_list.append(v)
          else:
            w = torch.randn(self.param_length+self.history_size).to(self.device)
            v = self._orthonormalization(w, v_list)
            v_list.append(v)
          w_prime = self.matrix_vector_product(v)
          alpha = w_prime.dot(v)
          alpha_list.append(alpha.cpu().item())
          w_tmp = w_prime - alpha * v
          w = w_tmp - beta * v_list[-2]

      # piece together tridiagonal matrix
      T = torch.zeros(num_iter, num_iter).to(self.device)
      for i in range(len(alpha_list)):
        T[i, i] = alpha_list[i]
        if i < len(alpha_list) - 1:
          T[i + 1, i] = beta_list[i]
          T[i, i + 1] = beta_list[i]

      eigenvalues, eigenvectors = torch.linalg.eig(T)

      eigen_list = eigenvalues.real
      weight_list = torch.pow(eigenvectors[0,:], 2) # only stores the square of first component of eigenvectors
      eigen_list_full.append(list(eigen_list.cpu().numpy()))
      weight_list_full.append(list(weight_list.cpu().numpy()))

    return eigen_list_full, weight_list_full

  """
  Helper function for normalizing a given vector. 
  """
  def _normalization(self, w): 
    return w / (w.norm() + 1e-6)

  """
  Helper function for orthonormalize given vector w with respect to a list of vectors (Gram–Schmidt + normalization). 
  """
  def _orthonormalization(self, w, v_list): 
    for v in v_list:
      w = w - w.dot(v) * v
    return self._normalization(w)