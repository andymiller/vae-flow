import urllib, gzip, os, pyprind
import numpy as np
import cPickle as pickle

def binarized_mnist():
    bmnist = np.load('BinaryMNIST/binarized_mnist.npz')
    return bmnist['arr_0'], bmnist['arr_1'], bmnist['arr_2']

def binarized_mnist_amat_to_numpy():
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in pyprind.prog_bar(lines)])
    with open(os.path.join('BinaryMNIST', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('BinaryMNIST', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('BinaryMNIST', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')
    return train_data, validation_data, test_data

if __name__ == '__main__':

    # download and convert to numpy arrays, pickle them
    for subdataset in subdatasets:
        filename = 'binarized_mnist_{}.amat'.format(subdataset)
        url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(subdataset)
        local_filename = os.path.join("BinaryMNIST", filename)
        urllib.urlretrieve(url, local_filename)

    bmnist = binarized_mnist_amat_to_numpy()
    np.savez_compressed('BinaryMNIST/binarized_mnist.npz', *bmnist)

