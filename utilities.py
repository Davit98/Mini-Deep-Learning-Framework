import torch
import math


def generate_disc_set(nb):
    data = torch.FloatTensor(nb, 2).uniform_(0, 1)
    target = torch.zeros(nb, 2)

    for i in range(nb):
        if ((data[i, 0] - 0.5)**2 + (data[i, 1] - 0.5)**2) <= 1 / (2 * math.pi):
            target[i, 1] = 1
        else:
            target[i, 0] = 1

    return data, target


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

    output = output.argmax(1)
    target = data_target.argmax(1)
    nb_errors += int((output != target).sum())

    return nb_errors
