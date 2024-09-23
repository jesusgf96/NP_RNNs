
import collections
import torch
import numpy as np
import torch.nn.functional as F
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import csv



def mackey_glass_sequence(sample_len=1000, tau=17, delta_t=10, n_samples=1):

    # Initial stuff
    history_len = tau * delta_t
    timeseries = 1.2
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



def generate_mackey_glass_data(n_batches=1, length=5000, predict_length=15, tau=17, washout=100, center=True):
    
    # Initial stuff
    X = np.stack(mackey_glass_sequence(
        sample_len=length+predict_length+washout, tau=tau, n_samples=n_batches*2), axis=1)
    X = X[washout:, :, :]
    
    # De-mean it
    if center:
        X -= np.mean(X)
    Y = X[:-predict_length, :, :]
    X = X[predict_length:, :, :]

    # Split into train and test
    return ((X[:, :n_batches], Y[:, :n_batches]), (X[:, n_batches:], Y[:, n_batches:]))



def generate_copy_data(delay=10, lenght=100, amp=10, batch_size=1):

    # Initial stuff
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



def generate_weather_data(delay_pred=1, batch_size=10, split_indx=33600):
    '''
        Climatological data of 1,600 U.S. locations from 2010 to 2013 acquired at https://www.ncei.noaa.gov/data/local-climatological-data/.
        This is a modification of the original usage of the dataset (exaplained in https://arxiv.org/pdf/2012.07436.pdf). 
    '''

    # Read data
    with open('data/WTH.csv', newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)))

    # Remove the repeated celsius variables (DryBulb, DewPoint, WetBulb)
    data = np.delete(data, [3,6,12], axis=1)

    # Separate weather features names and dates
    dates = data[1:, 0]
    features = data[0, 1:]
    data = data[1:,1:].astype(float)

    # Normalize data
    max_v = np.max(data, axis=0)
    min_v = np.min(data, axis=0)
    data = (data - min_v) / (max_v - min_v)

    # Separate in training and test set
    x_train = data[:split_indx]
    x_test = data[split_indx:]

    # Prepare the inputs and targets accordingly
    y_train = x_train[delay_pred:, 1]
    y_test = x_test[delay_pred:, 1]
    x_train = x_train[:-delay_pred]
    x_test = x_test[:-delay_pred]
    y_train = np.expand_dims(y_train, -1)
    y_test = np.expand_dims(y_test, -1)

    # Batch data
    indx_batch_train = int(len(x_train)/batch_size)
    indx_batch_test = int(len(x_test)/batch_size)
    x_train = np.stack([x_train[i*indx_batch_train:(i+1)*indx_batch_train] for i in range(batch_size)], axis=1)
    y_train = np.stack([y_train[i*indx_batch_train:(i+1)*indx_batch_train] for i in range(batch_size)], axis=1)
    x_test = np.stack([x_test[i*indx_batch_test:(i+1)*indx_batch_test] for i in range(batch_size)], axis=1)
    y_test = np.stack([y_test[i*indx_batch_test:(i+1)*indx_batch_test] for i in range(batch_size)], axis=1)

    # return dates, features, data
    return (x_train, y_train), (x_test, y_test)

