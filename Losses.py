from abc import ABC, abstractmethod
from numpy import log, sum


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
        return -(label * log(y) + (1 - label) * log(1 - y))

    def dloss(self, y, label):
        return (y - label) / (y * (1 - y))


class CategoricalCrossentropy(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, y, label):
        return -sum(label * log(y))

    def dloss(self, y, label):
        return -label / y
