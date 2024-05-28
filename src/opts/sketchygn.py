import torch
from torch.optim import Optimizer
from functools import reduce

def _armijo(f, x, gx, dx, t, alpha=0.1, beta=0.5):
    f0 = f(x, 0, dx)
    f1 = f(x, t, dx)
    while f1 > f0 + alpha * t * gx.dot(dx):
        t *= beta
        f1 = f(x, t, dx)
    return t

class SketchyGN(Optimizer):
    def __init__(self, params, lr=1.0, beta=0.999, rank=10, line_search_fn=None):
        defaults = dict(rank=rank, lr=lr, beta=beta,
                        line_search_fn=line_search_fn)
        self.rank = rank
        self.beta = beta
        self.line_search_fn = line_search_fn
        self.U = None
        self.S = None
        self.rho = None
        self.n_iters = 0
        self.n_sketch_upd = 0
        self.init_test_matrix = False

        super(SketchyGN, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError(
                "SketchyGN doesn't currently support per-parameter options (parameter groups)")

        if self.line_search_fn is not None and self.line_search_fn != 'armijo':
            raise ValueError("SketchyGN only supports Armijo line search")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        g = torch.cat([p.grad.view(-1)
                      for group in self.param_groups for p in group['params'] if p.grad is not None])
        g = g.detach()

        # one step update
        for group_idx, group in enumerate(self.param_groups):
            # calculate the preconditioned search direction
            UTg = torch.mv(self.U.t(), g)
            dir = torch.mv(self.U, (self.S + self.rho).reciprocal()
                           * UTg) + g / self.rho - torch.mv(self.U, UTg) / self.rho

            if self.line_search_fn == 'armijo':
                x_init = self._clone_param()

                def obj_func(x, t, dx):
                    self._add_grad(t, dx)
                    loss = float(closure())
                    self._set_param(x)
                    return loss

                # Use -dir for convention
                t = _armijo(obj_func, x_init, g, -dir, group['lr'])
            else:
                t = group['lr']

            # print('t: ', t)

            self.state[group_idx]['t'] = t

            # update parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = dir[ls:ls+np].view(p.shape)
                ls += np
                if p.grad is None:
                    continue
                p.data.add_(-dp, alpha=t)

        self.n_iters += 1

        return loss

    def update_sketch(self, v_flat_list, bsz):
        # Flatten the vector given by v
        # v_flat = torch.cat([component.view(-1) for component in v])
        p = v_flat_list[0].shape[0]

        # If the test matrix has not been initialized, initialize it
        if not self.init_test_matrix:
            self.Phi = torch.randn(
                (p, self.rank), device=self.param_groups[0]['params'][0].device) / (p ** 0.5)
            self.Phi = torch.linalg.qr(self.Phi, mode='reduced')[0]
            self.sketch = torch.zeros(
                (p, self.rank), device=self.param_groups[0]['params'][0].device)
            self.init_test_matrix = True

        # Update the sketch
        vvTPhi = torch.zeros((p, self.rank), device=self.param_groups[0]['params'][0].device)
        for v_flat in v_flat_list:
            PhiTv = torch.mv(self.Phi.t(), v_flat)
            vvTPhi += torch.outer(v_flat, PhiTv)

        # PhiTv = torch.mv(self.Phi.t(), v_flat)
        # vvTPhi = torch.outer(v_flat, PhiTv)
        self.sketch = self.beta * self.sketch + (1 - self.beta) * bsz * vvTPhi
        self.n_sketch_upd += 1

    def update_preconditioner(self):
        sketch_debiased = self.sketch / (1 - self.beta ** self.n_sketch_upd)

        # Calculate shift
        shift = torch.finfo(sketch_debiased.dtype).eps
        sketch_shifted = sketch_debiased + shift * self.Phi

        # Calculate Phi^T * sketch_debiased * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(self.Phi.t(), sketch_shifted)

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
            # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T
            C = torch.linalg.cholesky(
                torch.mm(eigvectors, torch.mm(torch.diag(eigs), eigvectors.t())))
            
        try:
            B = torch.linalg.solve_triangular(
                C, sketch_shifted.t(), upper=False, left=True)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except:
            B = torch.linalg.solve_triangular(C.to('cpu'), sketch_shifted.t().to(
                'cpu'), upper=False, left=True).to(C.device)
        _, S, UT = torch.linalg.svd(B, full_matrices=False)
        self.U = UT.t()
        self.S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        # Automatically set rho
        self.rho = self.S[-1]

        # print('S: ', self.S)
        # print('rho: ', self.rho)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data
