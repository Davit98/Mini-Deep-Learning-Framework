import torch
import matplotlib
import matplotlib.pyplot as plt
import math
from IPython import display
matplotlib.style.use('ggplot')
torch.set_grad_enabled(False)

from utilities import generate_disc_set, compute_nb_errors, get_batches
from loss import Loss, MSE
from module import DenseLayer, ReLU, Sequential, Tanh


train_size = 1000
test_size = 1000

train_data, train_target = generate_disc_set(train_size)
test_data, test_target = generate_disc_set(test_size)

plt.figure(1,figsize=(6,6))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target[:, 0].numpy(), edgecolors='none')
plt.title('Training Data')
plt.pause(1)
plt.show(block=False)

# plt.figure(figsize=(6,6))
# plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target[:, 0].numpy(), edgecolors='none')
# plt.title('Test Data')
# plt.show()

net_loss = MSE()

net = Sequential(DenseLayer(2, 50),
                 ReLU(),
                 DenseLayer(50, 50),
                 ReLU(),
                 DenseLayer(50, 2))
print(net)


def sgd(x, dx, config):
    for cur_layer_x, cur_layer_dx in zip(x, dx):
        for cur_x, cur_dx in zip(cur_layer_x, cur_layer_dx):

            cur_old_grad = config['learning_rate'] * cur_dx

            if cur_old_grad.shape[0] == 1:
                cur_x = cur_x.reshape(cur_old_grad.shape)

            cur_x.add_(-cur_old_grad)


def train_model(model, model_loss, train_data, train_target, lr=0.005, batch_size=1, n_epoch=60):
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

        print('Current train loss: {:.4f}'.format(loss.item()/k))

    return train_loss_history, test_loss_history


print('Training started...')
train_loss_history, test_loss_history = train_model(net, net_loss, train_data, train_target, n_epoch=50)


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
print("Number of errors on a train set: " + str(errors_train))
train_res = train_res.argmax(1)
train_res[train_res != train_target.argmax(1)] = 2

test_res = net.forward(test_data)
errors_test = compute_nb_errors(test_res, test_target)
print("Number of errors on a test set: " + str(errors_test))
test_res = test_res.argmax(1)
test_res[test_res != test_target.argmax(1)] = 2

plt.figure(3,figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_res, edgecolors="none")
plt.title(f'On Train Data, {errors_train} errors')

plt.subplot(1, 2, 2)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_res, edgecolors="none")
plt.title(f'On Test Data, {errors_test} errors')
plt.show()