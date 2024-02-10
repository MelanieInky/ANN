from Activation import *
from Convolution import convolution, make_correspondence_table


class Layer:
    def __init__(self):
        self.output_dim = None
        self.input_dim = None
        self.grad_w = None
        self.out = None
        self.w = None
        self.b = None
        self.grad_b = None
        self.activation = None
        self.layer_type = None
        self.is_last_layer = True

    @abstractmethod
    def forward(self, inp):
        pass

    def get_weights(self):
        if self.w is None:
            return None
        return self.w

    def get_output_dim(self):
        return self.output_dim

    def get_n_of_w_and_b(self):
        return np.prod(self.w.shape) + np.prod(self.b.shape)

    def initialize_w_and_b(self, mu, std_dev):
        rng = np.random.default_rng()
        if hasattr(self, "b"):
            self.b = rng.normal(mu, std_dev, self.b.shape)
        if hasattr(self, "w"):
            self.w = rng.normal(mu, std_dev, self.w.shape)

    def reset_all(self):
        # Reset all the output and gradients
        self.out.fill(0)
        self.grad_w.fill(0)
        self.grad_b.fill(0)

    def set_activation(self, activation):
        # Set the activation object
        if activation == "logistic":
            self.activation = Logistic()
        elif activation == "linear":
            self.activation = Linear()
        elif activation == "Relu" or activation == "ReLu" or activation == "relu":
            self.activation = ReLu()
        elif activation == "tanh" or activation == "Tanh":
            self.activation = Tanh()
        elif activation == "softmax" or activation == "Softmax":
            self.activation = SoftMax()
        else:
            raise NotImplementedError("Unknown activation function")

    def learn(self, learning_rate=0.01):
        if hasattr(self, "w"):
            self.w -= self.grad_w * learning_rate
        if hasattr(self, "b"):
            self.b -= self.grad_b * learning_rate

    def __str__(self):
        str_out = f"{self.layer_type} layer with\n"
        str_out += f"- Input dimensions: {self.input_dim}\n"
        str_out += f"- Output dimensions: {self.output_dim}\n"
        if hasattr(self, "w"):
            str_out += f"- Number of weights+bias: {self.get_n_of_w_and_b()}\n"
        return str_out


# DENSE LAYER ##############


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
        if self.is_last_layer:
            # We assume loss is categorical cross entropy here
            delta = dloss * self.activation.dphi_phi(self.out)
        else:
            delta = next_layer.delta @ next_layer.w
            delta *= self.activation.dphi_phi(self.out)

        self.delta = delta
        self.grad_w += np.outer(self.delta, inp)
        self.grad_b += self.delta
        pass


# Convolutional Layer #########################


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


# Flattening layer #######################


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


# TODO: pooling layer


if __name__ == "__main__":
    conv_layer = ConvolutionLayer((4, 4), (2, 2), activation="linear", stride=2)
    conv_layer.initialize_tables()
    A = np.arange(16).reshape((4, 4, 1))
    conv_layer.w[0] = np.ones((2, 2, 1))
    conv_layer.forward(A)
    conv_layer.out
