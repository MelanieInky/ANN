import numpy as np

from Layer import *
from Losses import *


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
            self.loss_object = CategoricalCrossentropy()
        elif loss == 'CrossEntropy' or 'Cross entropy':
            self.loss_object = CrossEntropy()
        elif loss == 'MSE' or 'mean squared error':
            self.loss_object = MeanSquaredError()
        else:
            raise NotImplementedError('Unknown loss')

    def set_loss_part(self, loss, output, label):
        """
        Compute the loss for a single input
        Args:
            label:
            output:
            loss:

        Returns:

        """
        self.loss += self.loss_object.loss(output, label)

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
        dloss = self.loss_object.dloss(self.output, label)
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

    def __str__(self) -> str:
        str_out = "Neural network with:\n"
        str_out += f"Input dimensions {self.input_dim}\n"
        str_out += f"Output dimensions {self.output_dim}\n"
        str_out += f"----- Layers: -----\n"
        for layer in self.layer_list:
            str_out += layer.__str__()
        return str_out

    def get_output(self):
        print(self.layer_list[-1].out)


if __name__ == "__main__":
    # Test the dense layer thingy
    X = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    Y = np.zeros((3, 3))
    Y[0, 0] = 1
    Y[1, 1] = 1
    Y[2, 2] = 1

    ann = NeuralNetwork(X, Y)
    # ann.add_dense_layer(3,'logistic')
    ann.add_dense_layer(3, "softmax")
    ann.initialize_weights()
    ann._learn_batch(3, 10000, 0.1)
    ann.forward_nn(X[0])
    ann.get_output()
    ann.forward_nn(X[1])
    ann.get_output()
    ann.forward_nn(X[2])
    ann.get_output()

    print(ann)
