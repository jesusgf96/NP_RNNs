
import collections
import torch
import numpy as np
import torch.nn.functional as F
from tensorflow.keras.datasets import mnist
import tensorflow as tf


def mackey_glass_sequence(sample_len=1000, tau=17, delta_t=10, seed=0, n_samples=1):
    # Adapted from https://github.com/mila-iqia/summerschool2015/blob/master/rnn_tutorial/synthetic.py
    '''
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    '''

    # Initial stuff
    history_len = tau * delta_t
    timeseries = 1.2
    np.random.seed(seed)
    samples = []

    # Iterate samples
    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        
        # Iterate timesteps
        inp = np.zeros((sample_len,1))
        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                             0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries

        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)
    return samples



def generate_mackey_glass_data(n_batches=1, length=5000, seed=0, predict_length=15, tau=17, washout=100, center=True):
    
    # Initial stuff
    X = np.stack(mackey_glass_sequence(
        sample_len=length+predict_length+washout, tau=tau,
        seed=seed, n_samples=n_batches*2), axis=1)
    X = X[washout:, :, :]
    
    # De-mean it
    if center:
        X -= np.mean(X)
    Y = X[:-predict_length, :, :]
    X = X[predict_length:, :, :]

    # Split into train and test
    return ((X[:, :n_batches], Y[:, :n_batches]), (X[:, n_batches:], Y[:, n_batches:]))



def generate_copy_data(delay=10, lenght=100, amp=10, batch_size=1, seed=0):

    # Initial stuff
    np.random.seed(seed)
    zeros_x = np.zeros((delay, batch_size))
    markers = (amp-1) * np.ones((lenght, batch_size))
    zeros_y = np.zeros((lenght+delay, batch_size))
    
    # Generate random sequence
    seq = np.random.randint(low=1, high=amp-1, size=(lenght, batch_size))

    # Concatenate with markers and control bit
    x_ = np.concatenate([seq, zeros_x, markers], axis=0)
    y_ = np.concatenate([zeros_y, seq], axis=0)

    # Expand to one-hot encoding
    x = F.one_hot(torch.tensor(x_, dtype=torch.int64), amp).float()
    y = torch.tensor(y_, dtype=torch.int64)
    y = F.one_hot(y, amp).float()
    return x, y



def generate_psMNIST_data(seed=0):

    # Load MNIST and normalize
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    # Flatten and add batch dim
    x_train = np.expand_dims(x_train.reshape((x_train.shape[0], -1, 1)), -1)
    x_test = np.expand_dims(x_test.reshape((x_test.shape[0], -1, 1)), -1)

    # Permutate pixels with same seed
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(x_train.shape[1])
    x_train = x_train[:, perm]
    x_test = x_test[:, perm]

    # One-hot encode the labels
    num_classes = 10
    y_train = np.expand_dims(tf.keras.utils.to_categorical(y_train, num_classes), 1)
    y_test = np.expand_dims(tf.keras.utils.to_categorical(y_test, num_classes), 1)
    return (x_train, y_train), (x_test, y_test)

