import torch
import matplotlib
import matplotlib.pyplot as plt
import math
matplotlib.style.use('ggplot')
torch.set_grad_enabled(False)

from Loss import Loss, MSE
from Module import DenseLayer, ReLU, Sequential, Tanh

def generate_disc_set(nb):
    data = torch.FloatTensor(nb, 2).uniform_(0, 1)
    target = torch.zeros(nb, 2)

    for i in range(nb):
        if (data[i, 0] - 0.5) * (data[i, 0] - 0.5) + \
                (data[i, 1] - 0.5) * (data[i, 1] - 0.5) <= 1 / (2 * math.pi):
            target[i, 1] = 1
        else:
            target[i, 0] = 1

    return data, target


train_size = 1000
test_size = 1000

train_data, train_target = generate_disc_set(train_size)
test_data, test_target = generate_disc_set(test_size)

# plt.figure(figsize=(6,6))
# plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target[:, 0].numpy(), edgecolors='none')
# plt.title('Training Data')
# plt.show()
#
#
# plt.figure(figsize=(6,6))
# plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target[:, 0].numpy(), edgecolors='none')
# plt.title('Test Data')
# plt.show()

net_loss = MSE()

net = Sequential()
net.add(DenseLayer(2, 25))
net.add(ReLU())
net.add(DenseLayer(25, 25))
net.add(ReLU())
net.add(DenseLayer(25, 2))
net.add(ReLU())

print(net)


def sgd(x, dx, config):
    for cur_layer_x, cur_layer_dx in zip(x, dx):
        for cur_x, cur_dx in zip(cur_layer_x, cur_layer_dx):

            cur_old_grad = config['learning_rate'] * cur_dx

            if cur_old_grad.shape[0] == 1:
                cur_x = cur_x.reshape(cur_old_grad.shape)

            cur_x.add_(-cur_old_grad)


def get_batches(X, Y, batch_size):
    n_samples = X.shape[0]

    # shuffle
    indices = torch.randperm(n_samples)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_idx = indices[start:end]

        yield X[batch_idx], Y[batch_idx]


def compute_nb_errors(output, data_target):
    nb_errors = 0

    output = output.max(1).indices
    target = data_target.max(1).indices
    nb_errors += int((output != target).sum())

    return nb_errors


optimizer_config = {'learning_rate': 0.005}
n_epoch = 60
batch_size = 64
loss_history = []

for i in range(n_epoch):
    loss = 0
    for i in range(0, 1000):
        net.zero_grad_params()

        # Forward
        pred = net.forward(train_data[i])
        loss += net_loss.forward(pred, train_target[i])

        # Backward
        lg = net_loss.backward(pred, train_target[i])
        net.backward(train_data[i], lg)

        # Update weights
        sgd(net.get_params(),
            net.get_grad_params(),
            optimizer_config)
    loss_history.append(loss)

    # Visualize
    plt.figure(figsize=(8, 6))

    plt.title("Training loss")
    plt.xlabel("#iteration")
    plt.ylabel("loss")
    plt.plot(loss_history, 'b')
    plt.show()

    print('Current loss: %f' % loss)

res = []
for i in range(0, 1000):
    res.append(net.forward(train_data[i]).view(1, -1))
res = torch.cat(res, 0)
print("Number of errors on a train set: " + str(compute_nb_errors(res, train_target)))

res = []
for i in range(0, 1000):
    res.append(net.forward(test_data[i]).view(1, -1))
res = torch.cat(res, 0)
print("Number of errors on a test set: " + str(compute_nb_errors(res, test_target)))
# net.forward(test_data)
#
# mse = MSE()
# print(mse.update_output(X_train, Y_train))
# print(mse.update_output(X_test, Y_test))
#
#
# x1, x2 = torch.meshgrid(torch.linspace(0, 1), torch.linspace(0, 1))
# x1_flat = x1.flatten()
# x2_flat = x2.flatten()
#
# x_full = torch.stack([x1_flat, x2_flat]).T
# c = torch.argmax(net.forward(x_full), axis=1).reshape(x1.shape)
#
# plt.figure(figsize=(6, 6))
# plt.scatter(x1, x2, c=c)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=torch.argmax(Y_train, axis=1), edgecolors="none", s=5)
# plt.title('On Train Data')
# plt.show()
#
# # %%
#
# plt.figure(figsize=(6, 6))
# plt.scatter(x1, x2, c=c)
# plt.scatter(X_test[:, 0], X_test[:, 1], c=torch.argmax(Y_test, axis=1), edgecolors="none", s=5)
# plt.title('On Test Data')
# plt.show()