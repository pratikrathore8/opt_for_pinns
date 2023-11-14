'''
Original paper: PyHessian: Neural Networks Through the Lens of the Hessian (https://arxiv.org/abs/1912.07145)
Original authors implementation: https://github.com/amirgholami/PyHessian

Code is adapted to work with PINNs. 
'''

import numpy as np
import torch
import math

# utility functions

def group_product(xs, ys):
  """
  the inner product of two lists of variables xs,ys
  :param xs:
  :param ys:
  :return:
  """
  return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def group_add(params, update, alpha=1):
  """
  params = params + update*alpha
  :param params: list of variable
  :param update: list of data
  :return:
  """
  for i, p in enumerate(params):
    params[i].data.add_(update[i] * alpha)
  return params

def normalization(v):
  """
  normalization of a list of vectors
  return: normalized vectors v
  """
  s = group_product(v, v)
  s = s**0.5
  s = s.cpu().item()
  v = [vi / (s + 1e-6) for vi in v]
  return v

def get_params_grad(model):
  """
  get model parameters and corresponding gradients
  """
  params = []
  grads = []
  for param in model.parameters():
    if not param.requires_grad:
      continue
    params.append(param)
    grads.append(0. if param.grad is None else param.grad + 0.)
  return params, grads

def get_params(model):
  """
  get model parameters
  """
  params = []
  for param in model.parameters():
    if not param.requires_grad:
      continue
    params.append(param)
  return params

def hessian_vector_product(gradsH, params, v):
  """
  compute the hessian vector product of Hv, where
  gradsH is the gradient at the current point,
  params is the corresponding variables,
  v is the vector.
  """
  hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True)
  return hv

def orthnormal(w, v_list):
  """
  make vector w orthogonal to each vector in v_list.
  afterwards, normalize the output w
  """
  for v in v_list:
    w = group_add(w, v, alpha=-group_product(w, v))
  return normalization(w)

# hessian class

class hessian():
  """
  The class used to compute :
    i) the top 1 (n) eigenvalue(s) of the neural network
    ii) the trace of the entire neural network
    iii) the estimated eigenvalue density
  """

  def __init__(self, model, pred_func, loss_func, data=None, dataloader=None, cuda=True, device='cuda'):
    """
    model: the model that needs Hessain information
    pred_func: the prediction function
    loss_func: the loss function
    data: a single batch of data, including inputs and its corresponding labels
    dataloader: the data loader including bunch of batches of data
    """

    # make sure we either pass a single batch or a dataloader
    assert (data != None and dataloader == None) or (data == None and dataloader != None)

    self.model = model.eval()  # make model is in evaluation model
    self.pred_func = pred_func
    self.loss_func = loss_func

    if data != None:
      self.data = data
      self.full_dataset = False
    else:
      self.data = dataloader
      self.full_dataset = True

    if cuda:
      self.device = device
    else:
      self.device = 'cpu'

    # pre-processing for single batch case to simplify the computation.
    if not self.full_dataset:
      self.x, self.t = self.data
      x_res, x_left, x_right, x_upper, x_lower = self.x
      t_res, t_left, t_right, t_upper, t_lower = self.t
        
      if self.device != 'cpu':
        
        x_res, t_res = x_res.cuda(device=self.device), t_res.cuda(device=self.device)
        x_left, t_left = x_left.cuda(device=self.device), t_left.cuda(device=self.device)
        x_right, t_right = x_right.cuda(device=self.device), t_right.cuda(device=self.device)
        x_upper, t_upper = x_upper.cuda(device=self.device), t_upper.cuda(device=self.device)
        x_lower, t_lower = x_lower.cuda(device=self.device), t_lower.cuda(device=self.device)
        
        self.x = (x_res, x_left, x_right, x_upper, x_lower)
        self.t = (t_res, t_left, t_right, t_upper, t_lower)

      # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
      outputs = self.pred_func(self.x, self.t, self.model)
      loss = self.loss_func(self.x, self.t, outputs)
      grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
    # this step is used to extract the parameters from the model
    self.params = get_params(self.model)
    self.gradsH = [gradient if gradient is not None else 0.0 for gradient in grad_tuple]

  def dataloader_hv_product(self, v):

    device = self.device
    num_data = 0  # count the number of datum points in the dataloader

    THv = [torch.zeros(p.size()).to(device) for p in self.params]  # accumulate result
    for x, t in self.data:
      self.model.zero_grad()
      
      x_res, x_left, x_right, x_upper, x_lower = x
      t_res, t_left, t_right, t_upper, t_lower = t
      
      x_res, t_res = x_res.to(device), t_res.to(device)
      x_left, t_left = x_left.to(device), t_left.to(device)
      x_right, t_right = x_right.to(device), t_right.to(device)
      x_upper, t_upper = x_upper.to(device), t_upper.to(device)
      x_lower, t_lower = x_lower.to(device), t_lower.to(device)
      
      x = (x_res, x_left, x_right, x_upper, x_lower)
      t = (t_res, t_left, t_right, t_upper, t_lower)
      
      tmp_num_data = x_res.size(0)

      outputs = self.pred_func(x, t, self.model)
      loss = self.loss_func(x, t, outputs)
      grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
      gradsH = [gradient if gradient is not None else 0.0 for gradient in grad_tuple]
      params = get_params(self.model)
      
      self.model.zero_grad()
      Hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False)
      THv = [THv1 + Hv1 * float(tmp_num_data) + 0. for THv1, Hv1 in zip(THv, Hv)]
      num_data += float(tmp_num_data)

    THv = [THv1 / float(num_data) for THv1 in THv]
    eigenvalue = group_product(THv, v).cpu().item()
    
    return eigenvalue, THv

  def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
    """
    compute the top_n eigenvalues using power iteration method
    maxIter: maximum iterations used to compute each single eigenvalue
    tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
    top_n: top top_n eigenvalues will be computed
    """

    assert top_n >= 1

    device = self.device

    eigenvalues = []
    eigenvectors = []

    computed_dim = 0

    while computed_dim < top_n:
      eigenvalue = None
      v = [torch.randn(p.size()).to(device) for p in self.params]  # generate random vector
      v = normalization(v)  # normalize the vector

      for i in range(maxIter):
        v = orthnormal(v, eigenvectors)
        self.model.zero_grad()

        if self.full_dataset:
          tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
        else:
          Hv = hessian_vector_product(self.gradsH, self.params, v)
          tmp_eigenvalue = group_product(Hv, v).cpu().item()

        v = normalization(Hv)

        if eigenvalue == None:
          eigenvalue = tmp_eigenvalue
        else:
          if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
            break
          else:
            eigenvalue = tmp_eigenvalue
      eigenvalues.append(eigenvalue)
      eigenvectors.append(v)
      computed_dim += 1

    return eigenvalues, eigenvectors

  def trace(self, maxIter=100, tol=1e-3):
    """
    compute the trace of hessian using Hutchinson's method
    maxIter: maximum iterations used to compute trace
    tol: the relative tolerance
    """

    device = self.device
    trace_vhv = []
    trace = 0.

    for i in range(maxIter):
      self.model.zero_grad()
      v = [torch.randint_like(p, high=2, device=device) for p in self.params]
      # generate Rademacher random variables
      for v_i in v:
        v_i[v_i == 0] = -1

      if self.full_dataset:
        _, Hv = self.dataloader_hv_product(v)
      else:
        Hv = hessian_vector_product(self.gradsH, self.params, v)
      trace_vhv.append(group_product(Hv, v).cpu().item())
      if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
        return trace_vhv
      else:
        trace = np.mean(trace_vhv)

    return trace_vhv

  def density(self, iter=100, n_v=1):
    """
    compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
    iter: number of iterations used to compute trace
    n_v: number of SLQ runs
    """

    device = self.device
    eigen_list_full = []
    weight_list_full = []

    for k in range(n_v):
      v = [torch.randint_like(p, high=2, device=device) for p in self.params]
      # generate Rademacher random variables
      for v_i in v:
        v_i[v_i == 0] = -1
      v = normalization(v)

      # standard lanczos algorithm initlization
      v_list = [v]
      w_list = []
      alpha_list = []
      beta_list = []
      ############### Lanczos
      for i in range(iter):
        self.model.zero_grad()
        w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
        if i == 0:
          if self.full_dataset:
            _, w_prime = self.dataloader_hv_product(v)
          else:
            w_prime = hessian_vector_product(self.gradsH, self.params, v)
          alpha = group_product(w_prime, v)
          alpha_list.append(alpha.cpu().item())
          w = group_add(w_prime, v, alpha=-alpha)
          w_list.append(w)
        else:
          beta = torch.sqrt(group_product(w, w))
          beta_list.append(beta.cpu().item())
          if beta_list[-1] != 0.:
            # We should re-orth it
            v = orthnormal(w, v_list)
            v_list.append(v)
          else:
            # generate a new vector
            w = [torch.randn(p.size()).to(device) for p in self.params]
            v = orthnormal(w, v_list)
            v_list.append(v)
          if self.full_dataset:
            _, w_prime = self.dataloader_hv_product(v)
          else:
            w_prime = hessian_vector_product(self.gradsH, self.params, v)
          alpha = group_product(w_prime, v)
          alpha_list.append(alpha.cpu().item())
          w_tmp = group_add(w_prime, v, alpha=-alpha)
          w = group_add(w_tmp, v_list[-2], alpha=-beta)

      T = torch.zeros(iter, iter).to(device)
      for i in range(len(alpha_list)):
        T[i, i] = alpha_list[i]
        if i < len(alpha_list) - 1:
          T[i + 1, i] = beta_list[i]
          T[i, i + 1] = beta_list[i]
      a_, b_ = torch.linalg.eig(T)

      eigen_list = a_
      weight_list = torch.pow(b_, 2)
      eigen_list_full.append(list(eigen_list.cpu().numpy()))
      weight_list_full.append(list(weight_list.cpu().numpy()))

    return eigen_list_full, weight_list_full