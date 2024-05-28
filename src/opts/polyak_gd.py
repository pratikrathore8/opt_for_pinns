from torch.optim import Optimizer

class PolyakGD(Optimizer):
    def __init__(self, params):
        super(PolyakGD, self).__init__(params, defaults={})

    def step(self, closure):
        # Compute Polyak step size
        loss = float(closure())
        grad_norm_sq = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_norm_sq += p.grad.norm()**2

        t = loss / grad_norm_sq

        # Take Polyak step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data -= t * p.grad