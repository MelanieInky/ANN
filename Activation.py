"""
This is where we store the activation function. 
Activation function are of the form y = phi(x). For any activation function, we need

- A method phi(x).
- The derivative phi'(x), expressed as either a function of x, or directly y.
"""

from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self):
        self.is_special = False

    @abstractmethod
    def phi(self, x):
        pass

    @abstractmethod
    def dphi_phi(self, y):
        pass


class Logistic(Activation):
    def __init__(self) -> None:
        super().__init__()

    def phi(self, x):
        return 1 / (1 + np.exp(-x))

    def dphi(self, x):
        pass

    def dphi_phi(self, y):
        return y * (1 - y)


class Tanh(Activation):

    def __init__(self) -> None:
        super().__init__()

    def phi(self, x):
        ex = np.exp(x)
        emx = np.exp(-x)
        return (ex - emx) / (ex + emx)

    def dphi_phi(self, y):
        return 1 - y ** 2


class ReLu(Activation):

    def __init__(self) -> None:
        super().__init__()

    def phi(self, x):
        return max(0, x)

    def dphi_phi(self, y):
        return 1 if y > 0 else 0


class Linear(Activation):

    def __init__(self) -> None:
        super().__init__()

    def phi(self, x):
        return x

    def dphi_phi(self, y):
        return 1


class SoftMax(Activation):
    def __init__(self):
        super().__init__()
        self.is_special = True

    def phi(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def dphi_phi(self, y):
        return y * (1 - y)

    def dphi_special(self, y, i, j):
        # Returns dy_i/da_j
        if i == j:
            return y[i] * (1 - y[i])
        return -y[i]*y[j]


if __name__ == '__main__':
    act = ReLu()
    print(act.dphi_phi(0.01))
