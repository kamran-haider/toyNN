import os
from numpy.testing import assert_array_almost_equal, assert_array_equal
from toynn.models import BasicDeepModel
from toynn.layers import *
from toynn.utils import *


def test_init_constant():
    """
    Tests if the network parameters are initialized correctly, when using constant scaling.
    """
    test_params = {'W1': np.array([[ 0.01788628,  0.0043651 ,  0.00096497, -0.01863493, -0.00277388],
       [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
       [-0.01313865,  0.00884622,  0.00881318,  0.01709573,  0.00050034],
       [-0.00404677, -0.0054536 , -0.01546477,  0.00982367, -0.01101068]]),
    'b1': np.array([[ 0.],
       [ 0.],
       [ 0.],
       [ 0.]]),
    'W2': np.array([[-0.01185047, -0.0020565 ,  0.01486148,  0.00236716],
       [-0.01023785, -0.00712993,  0.00625245, -0.00160513],
       [-0.00768836, -0.00230031,  0.00745056,  0.01976111]]),
    'b2': np.array([[ 0.],
       [ 0.],
       [ 0.]])}
    np.random.seed(3)
    layers = [Input(5), ReLU(4), ReLU(3)]
    initialize_weights(layers)
    for index, l in enumerate(layers[1:]):
        assert_array_almost_equal(l.weights, test_params["W" + str(index + 1)], decimal=8)
        assert_array_almost_equal(l.biases, test_params["b" + str(index + 1)], decimal=8)

def test_weighted_inputs():
    pass


def test_basic_deep():
    """
    A broad integration type test, which trains a BasicDeepModel and check if its costs are
    correct after 200 epochs.
    """
    np.random.seed(1)
    ref_costs = np.array([0.7717493284237686, 0.6720534400822914])
    training_data = os.path.abspath("toyNN/tests/test_datasets/train_catvnoncat.h5")
    test_data = os.path.abspath("toyNN/tests/test_datasets/test_catvnoncat.h5")
    train_x_orig, train_y, test_x_orig, test_y, classes = load_test_data(training_data, test_data)
    num_px = train_x_orig.shape[1]
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    input_layer_nodes = num_px * num_px * 3
    layers = [Input(input_layer_nodes), ReLU(20), ReLU(7), ReLU(5), Sigmoid(1)]

    nn = BasicDeepModel(train_x, train_y, layers, weight_initialization="custom")
    nn.fit(learning_rate=0.0075, n_epochs=200)
    assert_array_almost_equal(ref_costs, [nn.costs[0], nn.costs[100]], decimal=8)


