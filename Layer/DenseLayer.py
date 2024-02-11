import numpy as np

from Layer.Layer import Layer


class DenseLayer(Layer):
    def __init__(self, output_dim, input_dim, activation="logistic", bias=True):
        super().__init__()
        self.layer_type = "Dense"
        self.input_dim = (input_dim,)
        self.output_dim = (output_dim,)

        self.out = np.zeros(output_dim)
        self.bias = bias

        self.b = np.zeros(output_dim)
        self.w = np.zeros((output_dim, input_dim))
        self.grad_w = np.zeros((output_dim, input_dim))
        self.grad_b = np.zeros(output_dim)
        self.delta = np.zeros(output_dim)

        self.set_activation(activation)
        pass

    def forward(self, inp):
        """Forward the current layer with the input inp
        Returns self.h for the next layer
        Args:
            inp (np.ndarray): The 1d input from the last layer.
        """
        self.out = self.w @ inp + self.b
        self.out = self.activation.phi(self.out)
        return self.out

    def backward(self, next_layer, inp, dloss=None):
        """Use after a forward pass to update the gradients values of
        the weights leading to this layer, treating it as the hidden
        layer


        # Args:
            next_layer (Layer): The next layer (in the 'next_layer')
            is the output layer sense.
            inp (1d array): The inp, from a further layer.
        """

        # dE/dã_n = sum_m [d E/d a_m * d a_m/d out_n * d out_n / d ã_n]
        # We get the
        delta = np.zeros_like(dloss)
        if self.is_last_layer:
            delta = self.__last_layer_backward(delta, dloss)
        else:
            if self.activation.is_special:
                raise NotImplementedError('Special activation(softmax) in a layer that is not the last')
            delta = next_layer.delta @ next_layer.w
            delta *= self.activation.dphi_phi(self.out)

        self.delta = delta
        self.grad_w += np.outer(self.delta, inp)
        self.grad_b += self.delta
        pass

    def __last_layer_backward(self, delta, dloss):
        if not self.activation.is_special:
            delta = dloss * self.activation.dphi_phi(self.out)
        # SOFTMAX is a bit special since the activation depends on all nodes.
        else:
            for j in range(len(self.delta)):
                partial_deriv = np.zeros(len(self.delta))
                for i in range(len(self.delta)):
                    partial_deriv[i] = self.activation.dphi_special(self.out, i, j)
                delta[j] = np.sum(dloss * partial_deriv)
        return delta
