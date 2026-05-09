from graph_encoders.graph_encoder import GraphEncoder
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

from collections import Counter

class WL(GraphEncoder):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
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
            document = WeisfeilerLehmanHashing(g, self.wl_iterations, self.attributed, self.erase_base_features)
            documents.append(document)

        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        return documents

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

    def calc_coefficients(self, corpus):

        sparse_vector = np.zeros([len(corpus), self.n_vocab])

        i = 0
        for corpus in corpus:
            words = corpus.words

            words_count = Counter(corpus.words)
            j = 0
            for atom, _ in self.vocab:
                sparse_vector[i][j] = words_count[atom]
                j = j + 1

            i = i + 1

        return sparse_vector

    def generate_training_embeddings(self, graphs):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        self.vocab = self.create_vocab(documents)
        train_graph_embeddings = self.calc_coefficients(documents)
        return train_graph_embeddings

    def generate_inferencing_embeddings(self, graphs):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        infer_graph_embeddings = self.calc_coefficients(documents)
        return infer_graph_embeddings