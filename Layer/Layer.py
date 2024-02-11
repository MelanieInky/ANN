from Activation import *


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

    def set_weight(self, w_coordinate: tuple, w: float):
        self.w[w_coordinate] = w

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
