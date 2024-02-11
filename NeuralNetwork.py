import numpy as np

from Layer.DenseLayer import DenseLayer
from Layer.FlattenLayer import FlattenLayer
from Losses import *
import numpy as np


class NeuralNetwork:
    def __init__(self, X: np.ndarray, Y: np.ndarray, loss='CatCrossEntropy') -> None:
        self.output = None
        self.input = None
        self.loss = 0

        self.layer_list = []
        self.X = X
        self.Y = Y
        self.input_dim = X[0].shape
        self.output_dim = Y[0].shape
        self.current_input_dim = self.input_dim
        self.label_dim = self.output_dim  # Alias for output_dim
        self.n_layers = 0
        if loss == 'CatCrossEntropy' or 'Categorical cross entropy':
            self.loss_func = CategoricalCrossentropy()
        elif loss == 'CrossEntropy' or 'Cross entropy':
            self.loss_func = CrossEntropy()
        elif loss == 'MSE' or 'mean squared error':
            self.loss_func = MeanSquaredError()
        else:
            raise NotImplementedError('Unknown loss')

    def add_partial_loss(self, label):
        """
        Compute the loss for the current label and output
        """
        self.loss += self.loss_func.loss(self.output, label)

    def add_dense_layer(self, n_nodes, activation="linear"):
        if self.layer_list:
            self.layer_list[-1].is_last_layer = False
        # Check if the input is flat
        if len(self.current_input_dim) == 1:
            layer = DenseLayer(n_nodes, self.current_input_dim[0], activation)
            self.current_input_dim = (n_nodes,)
            self.layer_list.append(layer)
            self.n_layers += 1
        else:
            layer = FlattenLayer(self.current_input_dim)
            self.layer_list.append(layer)
            self.n_layers += 1
            self.current_input_dim = layer.output_dim
            self.add_dense_layer(n_nodes, activation)

    def forward_nn(self, inp):
        # Do a whole forward pass with a specific inp
        self.input = inp  # Keep the input in memory
        layer_input = inp
        if not self.layer_list:
            raise ValueError('No layers found')

        last_layer_out_dim = self.layer_list[-1].get_output_dim()
        if last_layer_out_dim != self.label_dim:
            raise ValueError('Mismatch in output and labels dimension')
        for layer in self.layer_list:
            out = layer.forward(layer_input)
            layer_input = out
        self.output = out

    def backward_nn(self, label):
        dloss = self.loss_func.dloss(self.output, label)
        if self.n_layers == 1:
            self.layer_list[0].backward(None, self.input, dloss)
        else:
            # Setting up the last layer
            layer_input = self.layer_list[-2].out
            self.layer_list[-1].backward(None, layer_input, dloss)
            for i in range(self.n_layers - 2, 0, -1):
                layer_input = self.layer_list[i - 1].out
                self.layer_list[i].backward(self.layer_list[i + 1], layer_input)
            self.layer_list[0].backward(self.layer_list[1], self.input)

    def _learn_nn(self, learning_rate=0.01):
        for layer in self.layer_list:
            layer.learn(learning_rate)

    def _learn_batch(self, batch_size, n_epochs=1000, learning_rate=0.01):
        # To be made better but it will do for now

        for epoch_nbr in range(n_epochs):
            self.reset_gradients()
            for i in range(batch_size):
                self.forward_nn(self.X[i])
                self.backward_nn(self.Y[i])
            self._learn_nn(learning_rate / batch_size)

    def _learn1(self, i):
        for n in range(100):
            self.forward_nn(self.X[i])
            self.backward_nn(self.Y[i])
            self._learn_nn()
            self.reset_gradients()

    def initialize_weights(self, mu=0, std_dev=0.1):
        for layer in self.layer_list:
            layer.initialize_w_and_b(mu, std_dev)

    def reset_gradients(self):
        for layer in self.layer_list:
            layer.reset_all()

    def reset_loss(self):
        self.loss = 0

    def compare_loss_derivative_finite_diff(self,layer_number:int, coordinate:tuple, mode='w', sample_number=0):
        """
        Compare the derivative of the loss wrt to a specific weight using backprop  and finite difference.
        Useful for double-checking if everything works correctly
        Args:
            sample_number: which training sample to use
            mode:str. Default to 'w'. 'w' for weight coordinate, 'b' for bias coordinate.
            layer_number:
            coordinate:

        Returns:

        """
        eps = 1e-6
        weights = self.layer_list[layer_number].get_weights()
        biases = self.layer_list[layer_number].get_biases()
        if mode == 'w' and len(coordinate) != weights.ndim:
            raise Exception("The dimension of the weight tensor and the coordinate do not match")
        elif mode == 'b' and len(coordinate) != biases.ndim:
            raise Exception("The dimension of the bias tensor and the coordinate do not match")

        if mode == 'w':
            w_or_b = weights[coordinate]  # The specific weight to change
        elif mode == 'b':
            w_or_b = biases[coordinate]
        else:
            raise Exception("Mode should either be 'w' or 'b' ")
        # First get the gradients with
        self.reset_gradients()
        self.reset_loss()
        self.forward_nn(self.X[sample_number])
        self.add_partial_loss(Y[sample_number])
        loss1 = self.loss
        self.backward_nn(self.Y[sample_number])
        if mode == 'w':
            derivative_bp = self.layer_list[layer_number].grad_w[coordinate]
            self.layer_list[layer_number].set_weight(coordinate, w_or_b + eps)
        else:
            derivative_bp = self.layer_list[layer_number].grad_b[coordinate]
            self.layer_list[layer_number].set_bias(coordinate, w_or_b + eps)
        self.reset_loss()
        self.forward_nn(self.X[sample_number])
        self.add_partial_loss(Y[sample_number])
        loss2 = self.loss
        derivative_fd = (loss2 - loss1) / eps
        print(f'Derivative with the finite difference: {derivative_fd}')
        print(f'Derivative with the backward: {derivative_bp}')

    def __str__(self) -> str:
        str_out = "Neural network with:\n"
        str_out += f"Input dimensions {self.input_dim}\n"
        str_out += f"Output dimensions {self.output_dim}\n"
        str_out += f"----- Layers: -----\n"
        for layer in self.layer_list:
            str_out += layer.__str__()
        return str_out

    def get_output(self):
        print(self.output)
        return self.output


if __name__ == "__main__":
    # Test the dense layer thingy
    X = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    Y = np.zeros((3, 3))
    Y[0, 0] = 1
    Y[1, 1] = 1
    Y[2, 2] = 1

    ann = NeuralNetwork(X, Y)
    ann.add_dense_layer(3,'logistic')
    ann.add_dense_layer(5,activation='tanh')
    ann.add_dense_layer(3, "softmax")
    ann.initialize_weights()
    ann._learn_batch(1, 1, 0.1)

    print(ann)

    ann.compare_loss_derivative_finite_diff(1, (2,),'b',sample_number=1)
