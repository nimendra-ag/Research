from graph_encoders.graph_encoder import GraphEncoder
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

from collections import Counter


class WL(GraphEncoder):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
            n_vocab: int = 1000,
            min_features: int = 50
    ):

        super().__init__(name="ImbalanceAwareWL")

        self.seed = 42
        self.vocab = None
        self.graph_embeddings = None
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features

        # fallback max vocab
        self.n_vocab = n_vocab

        # minimum features if thresholding becomes too strict
        self.min_features = min_features

    # =========================================================
    # WL HASH EXTRACTION
    # =========================================================

    def create_wl_hash(self, graph_list):

        documents = []

        for graph in graph_list:
            g = self._check_graph(graph)

            document = WeisfeilerLehmanHashing(
                g, self.wl_iterations, self.attributed, self.erase_base_features)

            documents.append(document)

        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        return documents

    # =========================================================
    # HELLINGER-STYLE DISCRIMINATIVE VOCAB SELECTION
    # =========================================================

    def create_vocab(self, corpus, labels):
        majority_df = Counter()
        minority_df = Counter()

        majority_graphs = 0
        minority_graphs = 0

        for doc, label in zip(corpus, labels):

            # unique subtree hashes in this graph
            # document frequency instead of raw counts
            unique_features = Counter(doc.words)

            if label == -1:

                majority_graphs += 1

                for feature in unique_features:
                    majority_df[feature] += 1

            else:

                minority_graphs += 1

                for feature in unique_features:
                    minority_df[feature] += 1

        # ---------------------------------------------
        # Build feature scores
        # ---------------------------------------------

        all_features = set(
            list(majority_df.keys()) +
            list(minority_df.keys())
        )

        scored_vocab = []

        for feature in all_features:

            # -----------------------------------------
            # Class-normalized frequencies
            # P(feature | class)
            # -----------------------------------------

            p_majority = (
                majority_df[feature] / max(majority_graphs, 1)
            )

            p_minority = (
                minority_df[feature] / max(minority_graphs, 1)
            )

            # -----------------------------------------
            # Hellinger-style discriminative score
            # -----------------------------------------

            # Simple HD-inspired distance
            discriminative_score = abs(
                np.sqrt(p_majority) - 
                np.sqrt(p_minority)
           )
            # -----------------------------------------
            # Overall usefulness
            # -----------------------------------------

            total_presence = (
                p_majority + p_minority
            )

            # # Final score
            # lambda_weight = 0.3

            # score = (
           #  total_presence
           #  + lambda_weight * discriminative_score
           # )
            score = total_presence * discriminative_score

            scored_vocab.append(
                (feature, score)
            )

        # ---------------------------------------------
        # Sort features by discriminative importance
        # ---------------------------------------------

        scored_vocab = sorted(
            scored_vocab,
            key=lambda x: x[1],
            reverse=True
        )

        # ----------------------------------------
        # Adaptive selection
        # ----------------------------------------

        scores = np.array([x[1] for x in scored_vocab])

        # Threshold = mean + std
        threshold = scores.mean() - scores.std()

        trimmed_vocab = [
            item for item in scored_vocab
            if item[1] >= threshold
        ]

        # fallback if too few selected
        print(f"selected {len(trimmed_vocab)} from the adaptive selection method")
        if len(trimmed_vocab) < 50:
            trimmed_vocab = scored_vocab[:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)

        return trimmed_vocab

    # =========================================================
    # FEATURE VECTOR CREATION
    # =========================================================

    def calc_coefficients(self, corpus):

        sparse_vector = np.zeros(
            [len(corpus), self.n_vocab]
        )

        for i, document in enumerate(corpus):

            words_count = Counter(document.words)

            for j, (feature, _) in enumerate(self.vocab):

                sparse_vector[i][j] = (
                    words_count[feature]
                )

        # ---------------------------------------------
        # Optional L2 normalization
        # Prevent large graphs dominating
        # ---------------------------------------------

        norms = np.linalg.norm(
            sparse_vector,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1

        sparse_vector = sparse_vector / norms

        return sparse_vector

    # =========================================================
    # TRAINING EMBEDDINGS
    # =========================================================

    def generate_training_embeddings(self, graphs, labels):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        self.vocab = self.create_vocab(documents, labels)
        train_graph_embeddings = self.calc_coefficients(documents)
        return train_graph_embeddings

    def generate_inferencing_embeddings(self, graphs):
        self._set_seed()
        documents = self.create_wl_hash(graphs)
        infer_graph_embeddings = self.calc_coefficients(
            documents
        )
        return infer_graph_embeddings