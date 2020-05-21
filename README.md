# Mini-Deep-Learning-Framework

This is a mini deep learning framework that uses only pytorch’s tensor operations and python’s standard math library.

#### Prerequisties

* pytorch
* math
* matplotlib

#### File descriptions

module.py - this is the main .py file encapsulating the key components for implementing a neural network. It contains the following classes: 

* *Module* - a backbone abstract class which defines the necessary methods for training a neural network.
* *Sequential* - a container class of modules which processes input data sequentially. Inherits *Module*. 
* *DenseLayer* - a class implementing fully-connected layer. Inherits *Module*.
* *ReLU*, *Tanh*, *Sigmoid*, *Softmax* activation functions. All of them inherit *Module*.

loss.py - this is a .py file implementing neural network loss functions. It contains the following classes: 

* *Loss* - an abstract class defining all the necessary methods that a neural network loss function should have.
* *MSE* - a class implementing mean squared error loss. Inherits from *Loss*
* *CrossEntropy* - a class implementing categorical cross-entropy loss. Inherits from *Loss*.

utilities.py - this is a .py file containig several utility methods such as generating the data for the given binary classification problem, computing the accuracy of the neural net, etc.

test.py - this is the main executable python file which trains and evaluates the two deep neural network models described in the report. 

#### How to run

To run the (default) model with the MSE loss: ```python3 test.py```\
To run the model with the softmax loss: ```pyhton3 test.py --loss softmax_loss```
