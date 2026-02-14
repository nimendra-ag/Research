from abc import ABC, abstractmethod
from ksvd import ApproximateKSVD

class DictLearner(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self):
        pass