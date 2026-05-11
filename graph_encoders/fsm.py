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

    def create_vocab(self, documents, labels):
        """
        NEW: Class-Aware Trimming.
        Allocates vocabulary slots equally among all classes to prevent
        majority classes from overshadowing minority features.
        """
        unique_classes = np.unique(labels)
        # Distribute the n_vocab limit equally among classes
        vocab_per_class = self.n_vocab // len(unique_classes)

        final_vocab_dict = {}

        for cls in unique_classes:
            # 1. Isolate documents belonging to the current class
            cls_docs = [doc for doc, label in zip(documents, labels) if label == cls]

            # 2. Count frequencies only within this class
            cls_counts = Counter()
            for doc in cls_docs:
                cls_counts.update(doc)

            # 3. Filter out rare noise within the class
            frequent_cls_subgraphs = {
                word: count for word, count in cls_counts.items() if count >= self.min_count
            }

            # 4. Sort and take the top N features for this specific class
            sorted_cls_vocab = sorted(frequent_cls_subgraphs.items(), key=lambda item: item[1], reverse=True)
            top_cls_features = sorted_cls_vocab[:vocab_per_class]

            # 5. Add to the global dictionary (keeping the highest count if there's overlap)
            for shape, count in top_cls_features:
                if shape not in final_vocab_dict:
                    final_vocab_dict[shape] = count
                else:
                    final_vocab_dict[shape] += count  # Accumulate global count for importance ranking

        # Sort the final merged vocabulary by global occurrence
        sorted_merged_vocab = sorted(final_vocab_dict.items(), key=lambda item: item[1], reverse=True)

        self.n_vocab = len(sorted_merged_vocab)
        return sorted_merged_vocab

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