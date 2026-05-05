from abc import ABC, abstractmethod
from karateclub.estimator import Estimator


class GraphEncoder(ABC, Estimator):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.embeddings = None

    @abstractmethod
    def generate_training_embeddings(self, graphs):
        pass

    @abstractmethod
    def generate_inferencing_embeddings(self, graphs):
        pass