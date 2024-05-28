import torch
from torch.optim import Optimizer

class ALRMAG(Optimizer):
    def __init__(self, params, beta=0.9):
        defaults = dict(beta=beta)
        self.d = None
        self.beta = beta
        super(ALRMAG, self).__init__(params, defaults)

    def step(self, closure):
        loss = float(closure())

        g = torch.cat([p.grad.view(-1) for group in self.param_groups for p in group['params'] if p.grad is not None])
        g = g.detach()

        if self.d is None:
            self.d = g
        else:
            self.d = self.beta * self.d + g

        # Compute Polyak step size
        t = loss / self.d.norm()**2

        # Take Polyak step
        for group in self.param_groups:
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = self.d[ls:ls+np].view(p.shape)
                ls += np
                if p.grad is None:
                    continue
                p.data.add_(-dp, alpha=t)