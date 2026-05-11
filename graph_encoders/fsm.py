from collections import Counter
import numpy as np
import networkx as nx
import pandas as pd
from graph_encoders.graph_encoder import GraphEncoder


class FSM(GraphEncoder):
    def __init__(
            self,
            radius: int = 1,
            n_vocab: int = 1000,
            min_count: int = 5
    ):
        super().__init__(name="FSM")
        self.radius = radius
        self.n_vocab = n_vocab
        self.min_count = min_count
        self.vocab = None
        self.embeddings = None

    def _get_subgraph_signature(self, sub_g):
        num_nodes = sub_g.number_of_nodes()
        num_edges = sub_g.number_of_edges()

        degrees = sorted([d for n, d in sub_g.degree()])
        deg_str = ",".join(map(str, degrees))

        first_node = list(sub_g.nodes())[0]
        if 'label' in sub_g.nodes[first_node]:
            labels = sorted([sub_g.nodes[n]['label'] for n in sub_g.nodes()])
            label_str = ",".join(labels)
            return f"N{num_nodes}_E{num_edges}_L[{label_str}]_D[{deg_str}]"

        return f"N{num_nodes}_E{num_edges}_D[{deg_str}]"

    def extract_subgraphs(self, target_graphs):
        documents = []
        for G in target_graphs:
            doc_words = []
            for node in G.nodes():
                ego_graph = nx.ego_graph(G, node, radius=self.radius, center=True, undirected=True)
                signature = self._get_subgraph_signature(ego_graph)
                doc_words.append(signature)
            documents.append(doc_words)
        return documents

    def create_vocab(self, documents, labels, min_support_ratio=0.05, n_vocab=1000):
        """
        Builds vocabulary based on Discriminative Scoring (Variance of Class Supports).
        Keeps shapes that show the biggest difference in frequency between classes.
        """
        unique_classes = np.unique(labels)

        # 1. Group documents by class and calculate class sizes
        class_docs = {cls: [] for cls in unique_classes}
        for doc, label in zip(documents, labels):
            # We use set(doc) so we only count Document Frequency (does the graph have it or not?)
            class_docs[label].append(set(doc))

        class_sizes = {cls: len(docs) for cls, docs in class_docs.items()}

        # 2. Count Document Frequency per class
        class_shape_counts = {cls: Counter() for cls in unique_classes}
        global_shapes = set()

        for cls, docs in class_docs.items():
            for doc in docs:
                class_shape_counts[cls].update(doc)
                global_shapes.update(doc)

        # 3. Calculate Discriminative Score for every single shape
        shape_scores = {}

        for shape in global_shapes:
            # Calculate the support rate (0.0 to 1.0) for this shape in each class
            support_rates = []
            max_support = 0.0  # Track the highest support in any single class

            for cls in unique_classes:
                # Support = (Number of graphs in this class with the shape) / (Total graphs in class)
                support = class_shape_counts[cls].get(shape, 0) / class_sizes[cls]
                support_rates.append(support)
                max_support = max(max_support, support)

            # Filter baseline noise: The shape MUST appear in at least X% of AT LEAST ONE class
            # This prevents us from keeping a shape that appears in 1 Active and 0 Inactives.
            if max_support < min_support_ratio:
                continue

            # The Discriminative Score is the statistical variance of its support rates.
            # High variance = Highly discriminative. Low variance = Equally common everywhere.
            score = np.var(support_rates)
            shape_scores[shape] = score

        # 4. Sort shapes by their discriminative score (highest first)
        sorted_discriminative_shapes = sorted(shape_scores.items(), key=lambda x: x[1], reverse=True)

        # 5. Take the top most discriminative features (up to n_vocab)
        final_vocab = sorted_discriminative_shapes[:n_vocab]

        self.n_vocab = len(final_vocab)
        print(f"Evaluated {len(global_shapes)} unique shapes.")
        print(f"Kept Top {self.n_vocab} highly discriminative shapes.")

        return final_vocab

    def calc_coefficients(self, documents):
        sparse_vector = np.zeros([len(documents), self.n_vocab])
        for i, doc in enumerate(documents):
            words_count = Counter(doc)
            for j, (subgraph_sig, _) in enumerate(self.vocab):
                sparse_vector[i][j] = words_count[subgraph_sig]
        return sparse_vector

    def generate_training_embeddings(self, graphs, labels):
        """Now requires labels to perform class-aware trimming."""
        documents = self.extract_subgraphs(graphs)

        # Pass the labels to the updated vocab creator
        self.vocab = self.create_vocab(documents, labels)
        raw_embeddings = self.calc_coefficients(documents)

        # REDUNDANCY / COLLINEARITY FILTER
        print(f"\nFiltering Collinear Subgraphs...")
        df = pd.DataFrame(raw_embeddings)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_indices = [column for column in upper.columns if any(upper[column] > 0.95)]

        self.embeddings = df.drop(columns=to_drop_indices).values
        self.vocab = [v for i, v in enumerate(self.vocab) if i not in to_drop_indices]
        self.n_vocab = len(self.vocab)

        print(f"Dropped {len(to_drop_indices)} redundant topologies. Final Vocab Size: {self.n_vocab}")

        return self.embeddings

    def generate_inferencing_embeddings(self, graphs):
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call generate_training_embeddings first.")

        documents = self.extract_subgraphs(graphs)
        return self.calc_coefficients(documents)