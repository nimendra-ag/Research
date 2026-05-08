from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from graph_encoders.graph_encoder import GraphEncoder


class FSM(GraphEncoder):
    def __init__(
            self,
            graphs,
            radius: int = 1,
            n_vocab: int = 1000,
            min_count: int = 5
    ):
        super().__init__(name="FSM")
        self.graphs = graphs
        self.radius = radius  # 1-hop or 2-hop subgraphs
        self.n_vocab = n_vocab
        self.min_count = min_count
        self.vocab = None
        self.graph_embeddings = None

    def _get_subgraph_signature(self, sub_g):
        """
        Creates an explainable string signature for a physical graphlet.
        Instead of a black-box hash, it outputs tangible metrics:
        e.g., "Nodes:4_Edges:3_Degs:1,1,2,2"
        """
        num_nodes = sub_g.number_of_nodes()
        num_edges = sub_g.number_of_edges()

        # Get sorted degree sequence to identify the topology shape
        degrees = sorted([d for n, d in sub_g.degree()])
        deg_str = ",".join(map(str, degrees))

        signature = f"N{num_nodes}_E{num_edges}_D[{deg_str}]"
        return signature

    def extract_subgraphs(self):
        """
        Extracts localized physical subgraphs (ego-networks) from all graphs.
        Returns a list of 'documents', where each document is a list of subgraph signatures.
        """
        documents = []
        for G in self.graphs:
            doc_words = []
            for node in G.nodes():
                # Extract the physical subgraph around the node
                ego_graph = nx.ego_graph(G, node, radius=self.radius, center=True, undirected=True)

                # Convert the physical shape into an explainable string word
                signature = self._get_subgraph_signature(ego_graph)
                doc_words.append(signature)

            documents.append(doc_words)
        return documents

    def create_vocab(self, documents):
        """
        Finds the most frequent subgraphs across the entire dataset.
        """
        global_counts = Counter()
        for doc in documents:
            global_counts.update(doc)

        # Filter by minimum support (min_count)
        frequent_subgraphs = {word: count for word, count in global_counts.items() if count >= self.min_count}

        # Sort by highest frequency and trim to n_vocab
        sorted_vocab = sorted(frequent_subgraphs.items(), key=lambda item: item[1], reverse=True)
        trimmed_vocab = sorted_vocab[:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, documents):
        """
        Builds the Y matrix for KSVD by counting vocabulary subgraphs in each graph.
        """
        sparse_vector = np.zeros([len(documents), self.n_vocab])

        for i, doc in enumerate(documents):
            words_count = Counter(doc)
            for j, (subgraph_sig, _) in enumerate(self.vocab):
                sparse_vector[i][j] = words_count[subgraph_sig]

        return sparse_vector

    def plot_feature_relationship(self, max_graphs=30, max_features=15):
        """
        Plots a heatmap showing the frequency of explainable subgraphs in each graph.
        """
        if self.graph_embeddings is None or self.vocab is None:
            print("Please run fit() before trying to plot.")
            return

        plot_data = self.graph_embeddings[:max_graphs, :max_features]
        # Vocabulary keys are now explainable physical signatures!
        feature_labels = [sig for sig, _ in self.vocab[:max_features]]
        graph_labels = [f"Graph {i}" for i in range(len(plot_data))]

        plt.figure(figsize=(14, 10))
        sns.heatmap(plot_data, cmap="YlGnBu", annot=False,
                    xticklabels=feature_labels, yticklabels=graph_labels)

        plt.title("Relationship Between Graphs and Frequent Subgraphs")
        plt.xlabel("Physical Subgraph Signatures")
        plt.ylabel("Graphs")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def generate_graph_embeddings(self):
        documents = self.extract_subgraphs()
        self.vocab = self.create_vocab(documents)
        self.graph_embeddings = self.calc_coefficients(documents)

    def fit(self):
        self.generate_graph_embeddings()
        return self.graph_embeddings