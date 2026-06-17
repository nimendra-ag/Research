from graph_encoders.graph_encoder import GraphEncoder
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from collections import Counter


class EdgeWL(GraphEncoder):

    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = True,
            erase_base_features: bool = True,
            n_vocab: int = 1000,
            min_features: int = 50
    ):

        super().__init__(name="ImbalanceAwareEdgeWL")

        self.seed = 42
        self.vocab = None
        self.graph_embeddings = None
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.min_features = min_features

    def create_wl_hash(self, graph_list):

        documents = []

        for graph in graph_list:

            g = self._check_graph(graph)

            # Initialize edge labels

            edge_labels = {}

            for u, v, data in g.edges(data=True):

                node_u = str(g.nodes[u].get("feature", "UNK"))
                node_v = str(g.nodes[v].get("feature", "UNK"))

                edge_type = str(data.get("bond_type", "SINGLE"))

                # canonical ordering of node labels
                pair = sorted([node_u, node_v])

                label = f"{pair[0]}-{edge_type}-{pair[1]}"

                edge_labels[tuple(sorted((u, v)))] = label

            graph_features = list(edge_labels.values())

            # WL iterations on edges

            for _ in range(self.wl_iterations):

                new_edge_labels = {}

                for (u, v), current_label in edge_labels.items():

                    neighbor_labels = []

                    # edges touching u
                    for nbr in g.neighbors(u):

                        if nbr == v:
                            continue

                        edge = tuple(sorted((u, nbr)))

                        if edge in edge_labels:
                            neighbor_labels.append(edge_labels[edge])

                    # edges touching v
                    for nbr in g.neighbors(v):

                        if nbr == u:
                            continue

                        edge = tuple(sorted((v, nbr)))

                        if edge in edge_labels:
                            neighbor_labels.append(edge_labels[edge])

                    neighbor_labels = sorted(neighbor_labels)

                    # aggregate neighboring edge labels
                    merged_label = (
                        current_label
                        + "_"
                        + "_".join(neighbor_labels)
                    )

                    # hash merged label
                    hashed_label = str(hash(merged_label))

                    new_edge_labels[tuple(sorted((u, v)))] = hashed_label

                edge_labels = new_edge_labels

                graph_features.extend(edge_labels.values())

            documents.append(
                TaggedDocument(
                    words=graph_features,
                    tags=[str(len(documents))]
                )
            )

        return documents

    def create_vocab(self, corpus, labels):

        majority_df = Counter()
        minority_df = Counter()

        majority_graphs = 0
        minority_graphs = 0

        for doc, label in zip(corpus, labels):

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

        all_features = set(
            list(majority_df.keys()) +
            list(minority_df.keys())
        )

        scored_vocab = []

        for feature in all_features:

            p_majority = majority_df[feature] / majority_graphs

            p_minority = minority_df[feature] / minority_graphs

            # HD-inspired discriminative score
            discriminative_score = abs(
                np.sqrt(p_majority) -
                np.sqrt(p_minority)
            )

            total_presence = p_majority + p_minority

            score = total_presence * discriminative_score

            scored_vocab.append((feature, score))

        # Sort features by discriminative importance
        scored_vocab = sorted(
            scored_vocab,
            key=lambda x: x[1],
            reverse=True
        )

        # Adaptive selection
        scores = np.array([x[1] for x in scored_vocab])

        threshold = scores.mean() - scores.std()

        trimmed_vocab = [
            item for item in scored_vocab
            if item[1] >= threshold
        ]

        # fallback if too few selected
        print(f"selected {len(trimmed_vocab)} from the adaptive selection method")

        if len(trimmed_vocab) < self.min_features:
            trimmed_vocab = scored_vocab[:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)


      # limit vocabulary size
        
        if len(trimmed_vocab) > 10000:
            trimmed_vocab = trimmed_vocab[:10000]
            print(f"trimmed vocab to 10000")

        self.n_vocab = len(trimmed_vocab)

        return trimmed_vocab
    
    def calc_coefficients(self, corpus):

        sparse_vector = np.zeros([len(corpus), self.n_vocab])

        for i, document in enumerate(corpus):

            words_count = Counter(document.words)

            for j, (feature, _) in enumerate(self.vocab):

                sparse_vector[i][j] = words_count[feature]

        return sparse_vector

    def generate_training_embeddings(self, graphs, labels):

        self._set_seed()

        documents = self.create_wl_hash(graphs)

        self.vocab = self.create_vocab(documents, labels)

        train_graph_embeddings = self.calc_coefficients(documents)

        return train_graph_embeddings

    def generate_inferencing_embeddings(self, graphs):

        self._set_seed()

        documents = self.create_wl_hash(graphs)

        infer_graph_embeddings = self.calc_coefficients(documents)

        return infer_graph_embeddings