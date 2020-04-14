import torch
import math


# Main class from which other layer classes inherit
class Module(object):
    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, inpt):
        return self.update_output(inpt)

    def backward(self, inpt, grad_output):
        self.update_grad_input(inpt, grad_output)
        self.acc_grad_params(inpt, grad_output)
        return self.grad_input

    def update_output(self, inpt):
        pass

    def update_grad_input(self, inpt, grad_output):
        # gradient with respect to input
        pass

    def acc_grad_params(self, inpt, grad_output):
        # gradient with respect to parameters
        pass

    def zero_grad_params(self):
        pass

    def get_params(self):
        return []

    def get_grad_params(self):
        return []

    def __repr__(self):
        return "Module"


# Container class
class Sequential(Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
        self.y = []

    def add(self, module):
        self.modules.append(module)

    def update_output(self, inpt):
        self.y = [inpt]

        for i in range(0, len(self.modules)):
            self.y.append(self.modules[i].forward(self.y[i]))

        self.output = self.y[-1]

        return self.output

    def backward(self, grad_output):
        n = len(self.modules)
        grad = grad_output

        for i in range(n - 1, -1, -1):
            grad = self.modules[i].backward(self.y[i], grad)

        self.grad_input = grad
        return self.grad_input

    def zero_grad_params(self):
        for module in self.modules:
            module.zero_grad_params()

    def get_params(self):
        return [x.get_params() for x in self.modules]

    def get_grad_params(self):
        return [x.get_grad_params() for x in self.modules]

    def __repr__(self):
        return "".join([str(x) + '\n' for x in self.modules])

    def __getitem__(self, x):
        return self.modules.__getitem__(x)


class DenseLayer(Module):
    def __init__(self, n_in, n_out):
        super(DenseLayer, self).__init__()

        # Initializing weights
        stdv = 1. / math.sqrt(n_in)
        self.W = torch.FloatTensor(n_out, n_in).uniform_(-stdv, stdv)
        self.b = torch.FloatTensor(n_out).uniform_(-stdv, stdv).view(-1, 1)

        self.gradW = torch.zeros_like(self.W)
        self.gradb = torch.zeros_like(self.b)

    def update_output(self, inpt):
        self.output = self.W.mm(inpt.view(-1, 1)) + self.b
        return self.output

    def update_grad_input(self, inpt, grad_output):
        self.grad_input = self.W.t().mm(grad_output)
        return self.grad_input

    def acc_grad_params(self, inpt, grad_output):
        self.gradW = grad_output.view(-1, 1).mm(inpt.view(1, -1))
        self.gradb = grad_output

    def zero_grad_params(self):
        self.gradW = torch.zeros_like(self.gradW)
        self.gradb = torch.zeros_like(self.gradb)

    def get_params(self):
        return [self.W, self.b]

    def get_grad_params(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        return 'Linear %d -> %d' % (s[1], s[0])


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def update_output(self, inpt):
        self.output = torch.max(inpt, torch.zeros_like(inpt))
        return self.output

    def update_grad_input(self, inpt, grad_output):
        inpt[inpt <= 0] = 0
        inpt[inpt > 0] = 1
        self.grad_input = grad_output * inpt
        return self.grad_input

    def __repr__(self):
        return "ReLU"


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def update_output(self, inpt):
        self.output = inpt.tanh()
        return self.output

    def update_grad_input(self, inpt, grad_output):
        self.grad_input = (1 - self.output ** 2) * grad_output
        return self.grad_input

    # Another way
    # def update_grad_input(self, inpt, grad_output):
    #     self.grad_input = 4 * (inpt.exp() + inpt.mul(-1).exp()).pow(-2) * grad_output
    #     return self.grad_input

    def __repr__(self):
        return "Tanh"
