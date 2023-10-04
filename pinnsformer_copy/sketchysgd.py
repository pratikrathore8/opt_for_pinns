import torch
from torch.optim import Optimizer
from torch.func import vmap
import math

class SketchySGD(Optimizer):
    """Implements SketchySGD. We assume that there is only one parameter group to optimize.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rank (int): sketch rank
        rho (float): regularization
        lr (float): learning rate
        weight_decay (float): weight decay parameter
        hes_update_freq (int): how frequently we should update the Hessian approximation
        momentum (float): momentum parameter
        proportional (bool): option to maintain lr to rho ratio, even when lr decays
        chunk_size (int): number of Hessian-vector products to compute in parallel
                          if set to None, binary search will be used to find the maximally allowed value
        line_search_fn (str): line search function to use (currently only 'backtracking', 'strong_wolfe' supported)
        verbose (bool): option to print out eigenvalues of Hessian approximation
    """
    def __init__(self, params, rank = 100, rho = 0.1, lr = 0.01, weight_decay = 0.0,
                 hes_update_freq = 100, momentum = 0.0, proportional = False, chunk_size = None,
                 line_search_fn = None, verbose = False):
        # initialize the optimizer    
        defaults = dict(rank = rank, rho = rho, lr = lr, weight_decay = weight_decay, 
                        hes_update_freq = hes_update_freq, momentum = momentum, proportional = proportional,
                        chunk_size = chunk_size, line_search_fn = line_search_fn)
        self.rank = rank
        self.hes_update_freq = hes_update_freq
        self.proportional = proportional
        self.chunk_size = chunk_size
        self.line_search_fn = line_search_fn
        self.verbose = verbose
        self.ratio = rho / lr
        self.n_iter = 0
        self.U = None
        self.S = None
        self.counter = 0
        self.momentum = momentum
        self.momentum_buffer = None
        self.verbose = verbose
        super(SketchySGD, self).__init__(params, defaults)

        if self.line_search_fn is not None: 
            if self.line_search_fn not in ['backtracking', 'strong_wolfe']:
                raise ValueError(f'Line search function {self.line_search_fn} not supported.')
            elif len(self.param_groups) != 1:
                raise ValueError('Line search only supported for a single parameter group.')
            elif self.momentum != 0.0:
                raise ValueError('Line search not supported with momentum.')
            # elif self.weight_decay != 0.0:
            #     raise ValueError('Line search not supported with weight decay.')

    def step(self, closure = None):
        loss = None
        grad_tuple = None
        if closure is not None:
            with torch.enable_grad():
                loss, grad_tuple = closure()

        # update Hessian approximation, if needed
        g = torch.cat([gradient.view(-1) for gradient in grad_tuple if gradient is not None])
        if self.n_iter % self.hes_update_freq == 0:
            params = []

            for group in self.param_groups:
                for p in group['params']:
                    params.append(p)

            # update preconditioner
            self._update_preconditioner(params, g)

        g = g.detach()

        # update momentum buffer
        if self.momentum_buffer is None: 
            self.momentum_buffer = g
        else:
            self.momentum_buffer = self.momentum * self.momentum_buffer + g

        # update parameters
        if self.line_search_fn is not None:             
            lr = self.param_groups[0]['lr']
            # Adjust rho to be proportional to lr, if necessary
            if self.proportional:
                rho = lr * self.ratio
            else:
                rho = self.param_groups[0]['rho']

            UTg = torch.mv(self.U.t(), self.momentum_buffer) 
            g_new = torch.mv(self.U, (self.S + rho).reciprocal() * UTg) + self.momentum_buffer / rho - torch.mv(self.U, UTg) / rho
            direction = -g_new

            # Print norms of g and g_new
            # print(f'Norm of g = {torch.norm(g)}')
            # print(f'Norm of g_new = {torch.norm(g_new)}')
            
            # Print cosine similarity between g and g_new
            # print(f'Cosine similarity between g and g_new = {torch.dot(g, g_new) / (torch.norm(g) * torch.norm(g_new))}')

            curr_params = self._clone_param(0)          
            def obj_func(params_curr, step_size, search_dir):
                # get new parameters
                self._add_grad(0, step_size, search_dir)
                loss, grad = closure()
                flat_grad = torch.cat([gradient.view(-1) for gradient in grad if gradient is not None]).detach()

                # reset parameters
                with torch.no_grad():
                    self._set_param(0, params_curr)

                return float(loss), flat_grad
            
            if self.line_search_fn == 'backtracking':
                t = self._backtracking(obj_func, loss, curr_params, g, direction, torch.dot(g, direction), lr)
            elif self.line_search_fn == 'strong_wolfe':
                t = self._strong_wolfe(obj_func, curr_params, lr, direction, loss, g, torch.dot(g, direction))

            # update model parameters
            self._add_grad(0, t, direction)

            # If t is not a tensor, convert it to a tensor
            if not torch.is_tensor(t):
                t = torch.tensor(t)

            # store step-size in state dict
            self.state[0]['step_size'] = t

        else:
        # one step update
            for group_idx, group in enumerate(self.param_groups):
                lr = group['lr']
                weight_decay = group['weight_decay']

                # Adjust rho to be proportional to lr, if necessary
                if self.proportional:
                    rho = lr * self.ratio
                else:
                    rho = group['rho']

                # compute gradient as a long vector
                # g = torch.cat([p.grad.view(-1) for p in group['params'] if p.grad is not None]) # only get gradients if they exist!
                # calculate the search direction by Nystrom sketch and solve
                UTg = torch.mv(self.U.t(), self.momentum_buffer) 
                g_new = torch.mv(self.U, (self.S + rho).reciprocal() * UTg) + self.momentum_buffer / rho - torch.mv(self.U, UTg) / rho
                
                ls = 0
                # update model parameters
                for p in group['params']:
                    gp = g_new[ls:ls+torch.numel(p)].view(p.shape)
                    ls += torch.numel(p)
                    p.data.add_(-lr * (gp + weight_decay * p.data)) # use weight decay (not same as L2 reg.)

                self.state[group_idx]['step_size'] = lr

        self.n_iter += 1

        return loss
    
    def _update_preconditioner(self, params, gradsH):
        p = gradsH.shape[0]
        # Generate test matrix (NOTE: This is transposed test matrix)
        Phi = (torch.randn(self.rank, p) / (p ** 0.5)).to(params[0].device)
        Phi = torch.linalg.qr(Phi.t(), mode = 'reduced')[0].t()

        # Calculate sketch (NOTE: This is transposed sketch)
        # Use binary search to find the maximally allowed chunk_size (only when chunk_size has not been set)
        if self.chunk_size is None: 
            self._set_chunk_size(params, gradsH, Phi)

        Y = self._hvp_vmap(gradsH, params)(Phi)

        # Calculate shift
        shift = torch.finfo(Y.dtype).eps * 100
        Y_shifted = Y + shift * Phi
        # Calculate Phi^T * H * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(Y_shifted, Phi.t())
        # Perform Cholesky, if fails, do eigendecomposition
        # The new shift is the abs of smallest eigenvalue (negative) plus the original shift
        try:
            C = torch.linalg.cholesky(choleskytarget)
        except:
            # eigendecomposition, eigenvalues and eigenvector matrix
            eigs, eigvectors = torch.linalg.eigh(choleskytarget)
            shift = shift + torch.abs(torch.min(eigs))
            # add shift to eigenvalues
            eigs = eigs + shift
            # print(f'Shift = {shift}')
            # print(f'Eigenvalues = {eigs}')
            # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T 
            C = torch.linalg.cholesky(torch.mm(eigvectors, torch.mm(torch.diag(eigs), eigvectors.T)))

        try: 
            B = torch.linalg.solve_triangular(C, Y_shifted, upper = False, left = True)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except: 
            B = torch.linalg.solve_triangular(C.to('cpu'), Y_shifted.to('cpu'), upper = False, left = True).to(C.device)
        _, S, UT = torch.linalg.svd(B, full_matrices = False) # B = V * S * U^T b/c we have been using transposed sketch
        self.U = UT.t()
        self.S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        if self.verbose: 
            print(f'Approximate eigenvalues = {self.S}')
            # print(f'Low-rank approximation (without rho) = {torch.mm(torch.mm(self.U, torch.diag(self.S)), self.U.t())}')

    def _set_chunk_size(self, params, gradsH, Phi, safety_margin=0.05, safety_margin_factor=0.95): 
        # start with the rank
        self.chunk_size = self.rank
        # set bounds for the search
        max_size = self.rank
        min_size = 1
        while(True): 
            # update lower bound if attempted computation was successful
            try: 
                self._hvp_vmap(gradsH, params)(Phi)
                min_size = self.chunk_size
                # search range has converged to a single point
                if max_size - min_size <= 1: 
                    # grab memory information
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    if free_mem / total_mem < safety_margin: 
                        min_size = int(safety_margin_factor * min_size)
                    # create some safety margin (e.g. 95% of the found size)
                    self.chunk_size = max(1, min_size)
                    torch.cuda.empty_cache()
                    break
            # update upper bound if attempted computation ran out of memory
            except RuntimeError as e:
                if str(e).startswith('CUDA out of memory.') and self.chunk_size > 1:
                    max_size = self.chunk_size
                    torch.cuda.empty_cache()
                # terminate if other runtime error occurred or chunk_size = 1 still ran out of memory
                else: 
                    raise e
            # halve the search range
            self.chunk_size = int(0.5 * (min_size + max_size))
        # report final chunk size
        print(f'SketchySGD: chunk size has been set to {self.chunk_size}.')

    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: hvp(grad_params, params, v), in_dims = 0, chunk_size=self.chunk_size)
    
    def _clone_param(self, group_idx):
        return [p.clone(memory_format=torch.contiguous_format) for p in self.param_groups[group_idx]['params']]
    
    def _add_grad(self, group_idx, step_size, update):
        # TODO: make this less hacky
        # If step_size is a tensor, convert it to a float
        if torch.is_tensor(step_size):
            step_size = step_size.item()
        offset = 0
        for p in self.param_groups[group_idx]['params']:
            numel = p.numel()
            p.data.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel

    def _set_param(self, group_idx, params_data):
        for p, pdata in zip(self.param_groups[group_idx]['params'], params_data):
            p.copy_(pdata)
    
    # Write a backtracking line search that uses the closure as the loss function
    # The line search function should take the following inputs: obj_func, loss_cur, parameters, gradient, search direction, initial step size, and backtracking parameters
    # The line search function should return the appropriate step size
    def _backtracking(self, obj_func, loss_cur, params, grad, search_dir, gtd, init_step_size, alpha=1e-4, beta=0.5):
        # initialize step size
        step_size = init_step_size
        # evaluate loss at current parameters
        loss_new, _ = obj_func(params, step_size, search_dir)

        # print(f'loss_new in backtracking ls = {loss_new}')

        # while loss at new parameters is greater than loss at current parameters plus sufficient decrease
        while math.isnan(loss_new) or loss_new > loss_cur + alpha * step_size * gtd:
            # update step size
            step_size *= beta
            # evaluate loss at new parameters
            loss_new, _ = obj_func(params, step_size, search_dir)

        return step_size

    def _cubic_interpolate(self, f1, f2, g1, g2, x1, x2, bounds=None): 
        if bounds is not None:
            xmin_bound, xmax_bound = bounds
        else:
            xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

        d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        d2_square = d1**2 - g1 * g2
            
        if d2_square >= 0:
            d2 = d2_square.sqrt()
            if x1 <= x2:
                min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            else:
                min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
            return min(max(min_pos, xmin_bound), xmax_bound)
        else:
            return (xmin_bound + xmax_bound) / 2.
        
    def _strong_wolfe(self, obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9,
                         tolerance_change=1e-9, max_ls=30): 
        d_norm = d.abs().max()
        g = g.clone(memory_format=torch.contiguous_format)
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals = 1
        gtd_new = g_new.dot(d)

        # print(f'gtd = {gtd}')
        
        if math.isinf(f_new) or math.isnan(f_new) or torch.isinf(gtd_new) or torch.isnan(gtd_new): 
            t = t / g.abs().sum()
            f_new, g_new = obj_func(x, t, d)
            gtd_new = g_new.dot(d)

        t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
        done = False
        ls_iter = 0
        while ls_iter < max_ls:
            if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
                bracket = [t_prev, t]
                bracket_f = [f_prev, f_new]
                bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
                bracket_gtd = [gtd_prev, gtd_new]
                break

            if abs(gtd_new) <= -c2 * gtd:
                bracket = [t]
                bracket_f = [f_new]
                bracket_g = [g_new]
                done = True
                break

            if gtd_new >= 0:
                bracket = [t_prev, t]
                bracket_f = [f_prev, f_new]
                bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
                bracket_gtd = [gtd_prev, gtd_new]
                break

            min_step = t + 0.01 * (t - t_prev)
            max_step = t * 10
            tmp = t
            t = self._cubic_interpolate(
                t_prev,
                f_prev,
                gtd_prev,
                t,
                f_new,
                gtd_new,
                bounds=(min_step, max_step))

            t_prev = tmp
            f_prev = f_new
            g_prev = g_new.clone(memory_format=torch.contiguous_format)
            gtd_prev = gtd_new
            f_new, g_new = obj_func(x, t, d)
            ls_func_evals += 1
            gtd_new = g_new.dot(d)
            ls_iter += 1

        if ls_iter == max_ls:
            bracket = [0, t]
            bracket_f = [f, f_new]
            bracket_g = [g, g_new]

        insuf_progress = False
        low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
        while not done and ls_iter < max_ls:
            if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change: break

            t = self._cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

            eps = 0.1 * (max(bracket) - min(bracket))
            if min(max(bracket) - t, t - min(bracket)) < eps:
                # interpolation close to boundary
                if insuf_progress or t >= max(bracket) or t <= min(bracket):
                    # evaluate at 0.1 away from boundary
                    if abs(t - max(bracket)) < abs(t - min(bracket)):
                        t = max(bracket) - eps
                    else:
                        t = min(bracket) + eps
                    insuf_progress = False
                else:
                    insuf_progress = True
            else:
                insuf_progress = False

            f_new, g_new = obj_func(x, t, d)
            ls_func_evals += 1
            gtd_new = g_new.dot(d)
            ls_iter += 1

            if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
                bracket[high_pos] = t
                bracket_f[high_pos] = f_new
                bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
                bracket_gtd[high_pos] = gtd_new
                low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
            else:
                if abs(gtd_new) <= -c2 * gtd:
                    done = True
                elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                    # old high becomes new low
                    bracket[high_pos] = bracket[low_pos]
                    bracket_f[high_pos] = bracket_f[low_pos]
                    bracket_g[high_pos] = bracket_g[low_pos]
                    bracket_gtd[high_pos] = bracket_gtd[low_pos]

                # new point becomes new low
                bracket[low_pos] = t
                bracket_f[low_pos] = f_new
                bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
                bracket_gtd[low_pos] = gtd_new

        t = bracket[low_pos]
        f_new = bracket_f[low_pos]
        g_new = bracket_g[low_pos]
        
        return t

def hvp(grad_params, params, v):
    Hv = torch.autograd.grad(grad_params, params, grad_outputs = v,
                              retain_graph = True)
    Hv = tuple(Hvi.detach() for Hvi in Hv)
    return torch.cat([Hvi.reshape(-1) for Hvi in Hv])

def group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def normalize(v):
    s = torch.sqrt(group_product(v, v))
    v = [x / (s + 1e-6) for x in v]
    return v