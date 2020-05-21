# Mini-Deep-Learning-Framework

This is a mini deep learning framework that uses only pytorch’s tensor operations and python’s standard math library.

#### File descriptions

module.py - this is the main .py file encapsulating the key components for implementing a neural network. It contains the following classes: 

* *Module* - a backbone abstract class which defines the necessary methods for training a neural network.
* *Sequential* - a container class of modules which processes input data sequentially. Inherits *Module*. 
* *DenseLayer* - a class implementing fully-connected layer. Inherits *Module*.
* *ReLU*, *Tanh*, *Sigmoid*, *Softmax* activation functions. All of them inherit *Module*.

loss.py - this is a .py file implementing neural network loss functions. It contains the following classes: 

* *Loss* - an abstract class defining all the necessary methods that a neural network loss function should have.
* *MSE* - a class implementing mean squared error loss. Inherits from *Loss*
* *CrossEntropy* - a class implementing categorical cross-entropy. Inherits from *Loss*.

utility.py - to be continued...


