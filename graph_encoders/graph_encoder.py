from abc import ABC, abstractmethod
from karateclub.estimator import Estimator

# abstract base class
class GraphEncoder(ABC, Estimator):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.embeddings = None
        self.graphs = None

    @abstractmethod
    def generate_graph_embeddings(self):
        pass

    @abstractmethod
    def fit(self):
        pass

