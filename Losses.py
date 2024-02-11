from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    def __init__(self):
        self.y = None
        self.label = None
        pass

    @abstractmethod
    def loss(self, y, label):
        pass

    @abstractmethod
    def dloss(self, y, label):
        pass


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y, label):
        return (y - label) ** 2

    def dloss(self, y, label):
        return 2 * (y - label)


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y, label):
        return -(label * np.log(y) + (1 - label) * np.log(1 - y))

    def dloss(self, y, label):
        return (y - label) / (y * (1 - y))


class CategoricalCrossentropy(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y, label):
        return -sum(label * np.log(y))

    def dloss(self, y, label):
        return -label / y
