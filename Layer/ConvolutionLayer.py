import numpy

from Convolution import convolution, make_correspondence_table
from Layer import Layer
import numpy as np


class ConvolutionLayer(Layer):
    def __init__(
            self,
            input_dim,
            kernel_dim,
            stride=1,
            n_filters=1,
            activation="logistic",
    ):

        super().__init__()
        self.kernel_tbl = None
        self.input = None
        self.in_out_tbl = None
        self.layer_type = "convolutional"
        # Setting the dimensions variables
        if len(input_dim) != len(kernel_dim):
            raise ValueError("The dimensions of the inp and kernel must be the same")
        if len(kernel_dim) == 3:
            self.k_x, self.k_y, self.k_z = kernel_dim
            self.i_x, self.i_y, self.i_z = input_dim
            if self.k_z != self.i_z:
                raise ValueError(
                    "The depth of the inp and the kernel must be the same"
                )
        elif len(kernel_dim) == 2:
            self.k_x, self.k_y = kernel_dim
            self.k_z = 1
            self.i_z = 1
            self.i_x, self.i_y = input_dim
        else:
            raise ValueError("Input must be 2d or 3d")

        self.n_filters = n_filters
        self.input_dim = (self.i_x, self.i_y, self.i_z)
        self.kernel_dim = (self.k_x, self.k_y, self.k_z)

        # Setting the stride
        if isinstance(stride, int):
            self.s_x = stride
            self.s_y = stride
        elif isinstance(stride, (tuple, list, np.ndarray)):
            if len(stride) != 2:
                raise ValueError("The stride must either be an int or of length 2")
            self.s_x = stride[0]
            self.s_y = stride[1]

        self.stride = (self.s_x, self.s_y)
        # Preset the output dimensions
        out_x = int((self.i_x - self.k_x) / self.s_x) + 1
        out_y = int((self.i_y - self.k_y) / self.s_y) + 1
        self.out = np.zeros((n_filters, out_x, out_y))

        # Set the bias vectors
        self.b = np.zeros(n_filters)
        self.grad_b = np.zeros_like(self.b)
        # Kernel weights, all in one big 4d table,
        # First dim is the filter number
        self.w = np.zeros((n_filters, self.k_x, self.k_y, self.k_z))
        self.grad_w = np.zeros_like(self.w)

        self.set_activation(activation)

    def forward(self, inp):
        for f in range(self.n_filters):
            kernel = self.w[f]
            self.out[f], _ = convolution(inp, kernel, self.stride)
            self.out[f] = self.activation.phi(self.out[f])
        self.input = inp
        return self.out

    def initialize_tables(self):
        # Initialize the look-up table with the correspondences between inp,
        # weight and outputs. Is used to get connected layers in backprop
        self.in_out_tbl, self.kernel_tbl = make_correspondence_table(
            self.input_dim, self.kernel_dim, self.stride
        )
        pass

    def backward(self, next_layer):

        pass


if __name__ == "__main__":
    conv_layer = ConvolutionLayer((4, 4), (2, 2), activation="linear", stride=2)
    conv_layer.initialize_tables()
    conv_layer.initialize_w_and_b(0,0.1)
    coord = (0,0,1,0)
    print(conv_layer.w[coord])
    A = np.arange(16).reshape((4, 4, 1))
    conv_layer.w[0] = np.ones((2, 2, 1))
    conv_layer.forward(A)

    conv_layer.out
