import torch
torch.set_grad_enabled(False)
import math
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from utilities import generate_disc_set, compute_nb_errors, get_batches, accuracy
from loss import Loss, MSE, CrossEntropy
from module import DenseLayer, ReLU, Sequential, Tanh, Sigmoid, SoftMax

def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', default='mse', help='loss function to use: mse or softmax_loss')

    return parser.parse_args()


def main(args):
    total_size = 2000
    train_size = 1000
    test_size = 1000

    data, target = generate_disc_set(total_size,random_state=1)

    train_data, train_target = data[:train_size], target[:train_size]
    test_data, test_target = data[test_size:], target[test_size:]

    colours = ['blue','green','red']

    def colour_labels(labels):
        return list(map(lambda x: colours[x], labels))

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=colour_labels(train_target.argmax(1)), edgecolors='none')
    plt.title('Train Data')
    plt.xlabel(r'$x_{1}$')
    plt.ylabel(r'$x_{2}$')

    plt.subplot(1, 2, 2)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=colour_labels(test_target.argmax(1)), edgecolors='none')
    plt.title('Test Data')
    plt.xlabel(r'$x_{1}$')
    plt.ylabel(r'$x_{2}$')

    plt.pause(1)
    plt.show(block=False)


    if args.loss == 'mse':
        net_loss = MSE()
        net = Sequential(DenseLayer(2, 25),
                         ReLU(),
                         DenseLayer(25, 25),
                         ReLU(),
                         DenseLayer(25, 25),
                         ReLU(),
                         DenseLayer(25, 2))
    else:
        net_loss = CrossEntropy()
        net = Sequential(DenseLayer(2, 25),
                         ReLU(), 
                         DenseLayer(25, 25), 
                         ReLU(), 
                         DenseLayer(25, 25),
                         ReLU(), 
                         DenseLayer(25, 2), 
                         SoftMax())


    def sgd(x, dx, config):
        for cur_layer_x, cur_layer_dx in zip(x, dx):
            for cur_x, cur_dx in zip(cur_layer_x, cur_layer_dx):

                cur_old_grad = config['learning_rate'] * cur_dx

                if cur_old_grad.shape[0] == 1:
                    cur_x = cur_x.reshape(cur_old_grad.shape)

                cur_x.add_(-cur_old_grad)


    def train_model(model, model_loss, train_data, train_target, lr=0.005, batch_size=1, n_epoch=50):
        optimizer_config = {'learning_rate': lr}
        train_loss_history = []
        test_loss_history = []
        
        for i in range(n_epoch):
            loss = 0
            
            k = 0
            for x_batch, y_batch in get_batches(train_data, train_target, batch_size):
                model.zero_grad_params()

                # Forward
                pred = model.forward(x_batch)
                loss += model_loss.forward(pred, y_batch)
                
                # Backward
                lg = model_loss.backward(pred, y_batch)
                model.backward(lg)
                
                # Update weights
                sgd(net.get_params(), 
                    net.get_grad_params(), 
                    optimizer_config)  
                k+=1
            
            train_loss_history.append(loss/k)
            
            test_pred = model.forward(test_data)
            test_loss = model_loss.forward(test_pred, test_target)
            test_loss_history.append(test_loss)

            print('#Epoch {}: current train loss = {:.4f}'.format(i+1,loss.item()/k))

        return train_loss_history, test_loss_history


    print('Training started...')
    train_loss_history, test_loss_history = train_model(net, net_loss, train_data, train_target, n_epoch=50)

    print('Final train loss: {:.4f}'.format(train_loss_history[-1]))
    print('Final test loss: {:.4f}'.format(test_loss_history[-1]))


    plt.figure(2,figsize=(8, 6))
    plt.title("Train and Test Loss")
    plt.xlabel("#Epochs")
    plt.ylabel("loss")
    plt.plot(train_loss_history, 'b')
    plt.plot(test_loss_history, 'r')
    plt.legend(['train loss', 'test loss'])
    plt.pause(1)
    plt.show(block=False)

    train_res = net.forward(train_data)
    errors_train = compute_nb_errors(train_res, train_target)
    print("Number of errors on the train set: " + str(errors_train))
    train_res = train_res.argmax(1)
    train_res[train_res != train_target.argmax(1)] = 2

    test_res = net.forward(test_data)
    errors_test = compute_nb_errors(test_res, test_target)
    print("Number of errors on the test set: " + str(errors_test))
    test_res = test_res.argmax(1)
    test_res[test_res != test_target.argmax(1)] = 2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=colour_labels(train_res), edgecolors='none')
    plt.xlabel(r'$x_{1}$')
    plt.ylabel(r'$x_{2}$')
    plt.title(f'Train Data, {errors_train} errors')

    plt.subplot(1, 2, 2)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=colour_labels(test_res), edgecolors='none')
    plt.xlabel(r'$x_{1}$')
    plt.ylabel(r'$x_{2}$')
    plt.title(f'Test Data, {errors_test} errors')

    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)



