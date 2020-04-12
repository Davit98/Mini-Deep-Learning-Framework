import torch


class Loss(object):
    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, inpt, target):
        return self.update_output(inpt, target)

    def backward(self, inpt, target):
        return self.update_grad_input(inpt, target)

    def update_output(self, inpt, target):
        return self.output

    def update_grad_input(self, inpt, target):
        return self.grad_input

    def __repr__(self):
        return "Loss"


class MSE(Loss):
    def __init__(self):
        super(MSE, self).__init__()

    def update_output(self, inpt, target):
        self.output = torch.sum((target.view(-1, 1) - inpt) ** 2) / inpt.shape[0]
        return self.output

    def update_grad_input(self, inpt, target):
        self.grad_input = -2 * (target.view(-1, 1) - inpt) / inpt.shape[0]
        return self.grad_input

    def __repr__(self):
        return "MSE"

