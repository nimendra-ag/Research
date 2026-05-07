from graph_encoders.graph_encoder import GraphEncoder
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

from collections import Counter, defaultdict


class WL(GraphEncoder):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = False,
            erase_base_features: bool = False,
            n_vocab: int = 1000,
            min_count: int = 5,
            epochs: int = 10
    ):
        super().__init__(name="WL")
        self.seed = 42
        self.vocab = None
        self.graph_embeddings = None
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.min_count = min_count
        self.epochs = epochs

    def create_wl_hash(self, graphs):
        documents = []
        for graph in graphs:
            g = self._check_graph(graph)
            document = WeisfeilerLehmanHashing(
                g,
                self.wl_iterations,
                self.attributed,
                self.erase_base_features
            )
            documents.append(document)

        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        return documents

    # ✅ NORMALIZED VOCAB CREATION
    def create_vocab(self, corpus, labels):

        class_counts = defaultdict(Counter)
        class_sizes = Counter(labels)

        # Count subtree frequencies per class
        for doc, label in zip(corpus, labels):
            class_counts[label].update(doc.words)

        # Normalize frequencies per class
        normalized_freq = Counter()

        for label in class_counts:
            for word, count in class_counts[label].items():
                normalized_freq[word] += count / class_sizes[label]

        # Sort and trim vocabulary
        sorted_vocab = sorted(
            normalized_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )

        trimmed_vocab = sorted_vocab[:self.n_vocab]
        self.n_vocab = len(trimmed_vocab)

        return trimmed_vocab

    # ✅ NORMALIZED FEATURE VECTORS
    def calc_coefficients(self, corpus):

        sparse_vector = np.zeros([len(corpus), self.n_vocab])

        for i, doc in enumerate(corpus):
            words_count = Counter(doc.words)
            total = sum(words_count.values())

            for j, (atom, _) in enumerate(self.vocab):
                if total > 0:
                    sparse_vector[i][j] = words_count[atom] / total
                else:
                    sparse_vector[i][j] = 0

        return sparse_vector

    # ✅ TRAINING (now requires labels)
    def generate_training_embeddings(self, graphs, labels):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        self.vocab = self.create_vocab(documents, labels)
        train_graph_embeddings = self.calc_coefficients(documents)
        return train_graph_embeddings

    # ✅ INFERENCE (no labels needed)
    def generate_inferencing_embeddings(self, graphs):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        infer_graph_embeddings = self.calc_coefficients(documents)
        return infer_graph_embeddings