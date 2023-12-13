from torch.optim import Adam, LBFGS, Optimizer

class Adam_LBFGS(Optimizer):
    def __init__(self, params, switch_epoch, adam_params, lbfgs_params):
        defaults = dict(switch_epoch=switch_epoch, adam_params=adam_params, lbfgs_params=lbfgs_params)

        self.switch_epoch = switch_epoch
        self.adam = Adam(params, **adam_params)
        self.lbfgs = LBFGS(params, **lbfgs_params)

        super(Adam_LBFGS, self).__init__(params, defaults)

        self.state['epoch'] = 0


    def step(self, closure=None):
        if self.state['epoch'] < self.switch_epoch:
            self.adam.step(closure)
        else:
            self.lbfgs.step(closure)
            if self.state['epoch'] == self.switch_epoch:
                print(f'Switching to LBFGS optimizer at epoch {self.state["epoch"]}')

        self.state['epoch'] += 1