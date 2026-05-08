from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

from graph_encoders.graph_encoder import GraphEncoder


class WL(GraphEncoder):
    def __init__(
            self,
            graphs,
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
        self.graphs = graphs
        self.graph_embeddings = None
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.min_count = min_count
        self.epochs = epochs

    def create_wl_hash(self):
        documents = []
        for graph in self.graphs:
            g = self._check_graph(graph)
            document = WeisfeilerLehmanHashing(g, self.wl_iterations, self.attributed, self.erase_base_features)
            documents.append(document)

        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        return documents

    def plot_feature_relationship(self, max_graphs=30, max_features=30):
        """
        Plots a heatmap showing the frequency of features in each graph.
        """
        if self.graph_embeddings is None or self.vocab is None:
            print("Please run fit() before trying to plot.")
            return

        # Slice the data to prevent the plot from becoming an unreadable blob
        plot_data = self.graph_embeddings[:max_graphs, :max_features]

        # Extract the WL hashes (atoms) for the X-axis labels
        feature_labels = [atom for atom, _ in self.vocab[:max_features]]

        # Create Graph labels (tags) for the Y-axis labels
        graph_labels = [f"Graph {i}" for i in range(len(plot_data))]

        # Create the plot
        plt.figure(figsize=(14, 10))
        sns.heatmap(plot_data, cmap="YlGnBu", annot=False,
                    xticklabels=feature_labels, yticklabels=graph_labels)

        plt.title("Relationship Between Graphs and WL Features")
        plt.xlabel("WL Features (Hashes)")
        plt.ylabel("Graphs")

        # Rotate x-axis labels so the long hashes don't overlap
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def create_vocab(self, corpus):
        d2v_model = Doc2Vec(vector_size=self.n_vocab, min_count=self.min_count, epochs=self.epochs)

        # d2v_model.build_vocab(train_corpus)
        total_words, corpus_count = d2v_model.scan_vocab(
            corpus_iterable=corpus, corpus_file=None,
            progress_per=10000, trim_rule=None
        )
        d2v_model.corpus_count = corpus_count
        d2v_model.corpus_total_words = total_words
        d2v_model.prepare_vocab(update=False, keep_raw_vocab=True, trim_rule=None)

        sorted_vocab = (sorted(d2v_model.raw_vocab.items(), key=lambda item: item[1], reverse=True))

        trimmed_vocab = sorted_vocab[0:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    # def calc_coefficients(self, corpus):
    #
    #     sparse_vector = np.zeros([len(corpus), self.n_vocab])
    #
    #     i = 0
    #     for corpus in corpus:
    #         words = corpus.words
    #
    #         words_count = Counter(corpus.words)
    #         j = 0
    #         for atom, _ in self.vocab:
    #             sparse_vector[i][j] = words_count[atom]
    #             j = j + 1
    #
    #         i = i + 1
    #
    #     return sparse_vector

    def calc_coefficients(self, corpus_docs):
        sparse_vector = np.zeros([len(corpus_docs), self.n_vocab])

        for i, doc in enumerate(corpus_docs):
            words_count = Counter(doc.words)
            for j, (atom, _) in enumerate(self.vocab):
                sparse_vector[i][j] = words_count[atom]

        return sparse_vector

    def generate_graph_embeddings(self):
        self._set_seed()
        documents = self.create_wl_hash()
        self.vocab = self.create_vocab(documents)
        self.graph_embeddings = self.calc_coefficients(documents)


    def fit(self):
        self.generate_graph_embeddings()
        return self.graph_embeddings
