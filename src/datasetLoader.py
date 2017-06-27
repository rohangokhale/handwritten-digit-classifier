import gzip
import cPickle
import numpy as np

def load_data():
    """Opens the file containing the MNIST data set and returns it in a
    format to be modified by the load_data_wrapper function.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """The load_data_wrapper functions returns a tuple containing the
    training data, validation data, and test data. The format of the
    data returned from load_data is not the most convenient, so here it
    is changed to play nicely with the network and its functions produced
    in the network class. 

    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Function to take a single digit j (the expected value of a
    handwritten digit) and return a 10-d vector with a zero in every
    place except the j-unit. This is how the digit is represented in
    vector form in the network class. 
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
