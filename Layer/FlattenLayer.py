import numpy

from Layer.Layer import Layer


class FlattenLayer(Layer):
    def __init__(self, input_layer_dim):
        super().__init__()
        self.delta = None
        self.output_dim = (np.prod(input_layer_dim),)  # in a tuple for consistency
        self.out = np.zeros(self.output_dim[0])
        self.input_dim = input_layer_dim
        self.layer_type = "Flattening"

    def forward(self, inp):
        self.out = inp.flatten()
        return self.out

    def backward(self, next_layer):
        self.delta = next_layer.delta.reshape(self.input_dim)
