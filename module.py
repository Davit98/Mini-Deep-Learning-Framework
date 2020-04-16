import torch
import math


class Module(object):
    """
    An abstract class which defines the necessary methods for training a neural network.
    """

    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, inpt):
        """
        Forward pass.
        """
        return self.update_output(inpt)

    def backward(self, inpt, grad_output):
        """
        Backward pass.
        """
        self.update_grad_input(inpt, grad_output)
        self.acc_grad_params(inpt, grad_output)
        return self.grad_input

    def update_output(self, inpt):
        """
        Compute the output using the given input and module's parameters.
        """
        pass

    def update_grad_input(self, inpt, grad_output):
        """
        Compute the gradient of the module with respect to its input.
        """
        pass

    def acc_grad_params(self, inpt, grad_output):
        """
        Compute the gradient of the module with respect to its parameters.
        """
        pass

    def zero_grad_params(self):
        """
        Zero grad parameters.
        """
        pass

    def get_params(self):
        """
        Return a list of the parameters. 
        If the module does not have parameters, return an empty list.
        """
        return []

    def get_grad_params(self):
        """
        Return a list of the gradents with respect to the parameters.
        If the module does not have parameters, return an empty list.
        """
        return []

    def __repr__(self):
        return "Module"


class Sequential(Module):
    """
    A container class of modules which processes input data sequentially.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
        self.y = []

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.modules = []
        self.y = []
        for module in args:
            self.modules.append(module)

    def add(self, module):
        """
        Add a module to the container.
        """
        self.modules.append(module)

    def update_output(self, inpt):
        """
        Forward input data sequentially by each module.
        """
        self.y = [inpt]

        for i in range(0, len(self.modules)):
            self.y.append(self.modules[i].forward(self.y[i]))

        self.output = self.y[-1]

        return self.output

    def backward(self, grad_output):
        """
        Backward output of the network sequentially through each module.
        """
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
    """
    A class implementing fully-connected layer.
    Accepts 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(DenseLayer, self).__init__()

        # Initializing weights (like Pytorch)
        stdv = 1. / math.sqrt(n_in)
        self.W = torch.FloatTensor(n_out, n_in).uniform_(-stdv, stdv)
        self.b = torch.FloatTensor(n_out).uniform_(-stdv, stdv)

        self.gradW = torch.zeros_like(self.W)
        self.gradb = torch.zeros_like(self.b)

    def update_output(self, inpt):
        self.output = inpt.mm(self.W.T) + self.b
        return self.output

    def update_grad_input(self, inpt, grad_output):
        self.grad_input = grad_output.mm(self.W)
        return self.grad_input

    def acc_grad_params(self, inpt, grad_output):
        for grad, inp in zip(grad_output, inpt):
            self.gradW += grad.unsqueeze(1) * inp / grad_output.shape[0]
        self.gradb = grad_output.sum(axis=0) / grad_output.shape[0]

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
        self.output = 2/(1+torch.exp(-2*inpt)) - 1
        return self.output

    def update_grad_input(self, inpt, grad_output):
        self.grad_input = (1 - self.output ** 2) * grad_output
        return self.grad_input

    def __repr__(self):
        return "Tanh"


class Sigmoid(Module):
    def __init__(self):
         super(Sigmoid, self).__init__()
    
    def update_output(self, inpt):
        self.output = 1/(1+torch.exp(-inpt))
        return self.output
    
    def update_grad_input(self, inpt, grad_output):
        self.grad_input = grad_output*self.output*(1-self.output)
        return self.grad_input

    def __repr__(self):
        return "Sigmoid"


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def update_output(self, inpt):
        x = torch.exp(inpt - torch.max(inpt))
        self.output = x / (x.sum(axis=1).reshape(-1,1))
        return self.output
    
    def update_grad_input(self, inpt, grad_output): 
        self.grad_input = self.output * (grad_output - torch.sum(self.output*grad_output,axis=1).reshape(inpt.shape[0],1))  
        return self.grad_input
    
    def __repr__(self):
        return "SoftMax"
