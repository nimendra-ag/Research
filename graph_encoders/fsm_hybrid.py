import multiprocessing as mp
from collections import Counter
import numpy as np
import networkx as nx
import igraph as ig
import pandas as pd

from graph_encoders.graph_encoder import GraphEncoder


# =====================================================================
# STANDALONE WORKER FUNCTION (For Multiprocessing)
# =====================================================================
def _process_single_graph_hybrid(args):
    nx_g, radius = args

    # Safely convert NetworkX nodes to 0-indexed contiguous integers just in case
    nx_g = nx.convert_node_labels_to_integers(nx_g)

    # 1. Convert NetworkX graph to iGraph
    ig_g = ig.Graph(n=nx_g.number_of_nodes(), edges=list(nx_g.edges()), directed=False)

    # 2. Extract Labels (if they exist from the NCI dataset)
    has_labels = False
    if len(nx_g.nodes) > 0 and 'label' in nx_g.nodes[0]:
        has_labels = True
        labels = [nx_g.nodes[i]['label'] for i in range(nx_g.number_of_nodes())]
        ig_g.vs['label'] = labels

    doc_words = []

    for i in range(ig_g.vcount()):
        # Extract localized ego-network using fast C-backend
        neighborhood = ig_g.neighborhood(vertices=i, order=radius)
        sub_g = ig_g.subgraph(neighborhood)

        # 3. Exact Canonical Labeling (The BLISS Algorithm)
        if has_labels:
            # Map string labels to integer colors so BLISS can understand them
            unique_labels = sorted(list(set(sub_g.vs['label'])))
            label_to_color = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            colors = [label_to_color[lbl] for lbl in sub_g.vs['label']]

            # Permute graph while preserving the chemical label topology
            canon_perm = sub_g.canonical_permutation(color=colors)
        else:
            canon_perm = sub_g.canonical_permutation()

        canon_g = sub_g.permute_vertices(canon_perm)

        # 4. Generate the exact structural signature
        edges_tuple = tuple(sorted([tuple(sorted(e)) for e in canon_g.get_edgelist()]))

        if has_labels:
            # The chemical labels are now sorted in exact canonical order!
            canon_labels = ",".join(canon_g.vs['label'])
            signature = f"N{canon_g.vcount()}_E{canon_g.ecount()}_L[{canon_labels}]_E{edges_tuple}"
        else:
            # Fallback for datasets like Reddit without labels
            degrees = sorted(canon_g.degree())
            deg_str = ",".join(map(str, degrees))
            signature = f"N{canon_g.vcount()}_E{canon_g.ecount()}_D[{deg_str}]_E{edges_tuple}"

        doc_words.append(signature)

    return doc_words


# =====================================================================
# THE HYBRID GRAPH ENCODER
# =====================================================================
class FSM_Hybrid(GraphEncoder):
    def __init__(
            self,
            radius: int = 1,
            n_vocab: int = 1000,
            min_count: int = 5,
            num_workers: int = None
    ):
        super().__init__(name="FSM_Hybrid")
        self.radius = radius
        self.n_vocab = n_vocab
        self.min_count = min_count
        self.vocab = None
        self.embeddings = None

        # Use provided cores, or max out the system
        self.num_workers = num_workers if num_workers else mp.cpu_count()

    def extract_subgraphs_parallel(self, target_graphs):
        print(f"Extracting subgraphs on {self.num_workers} CPU cores...")
        tasks = [(g, self.radius) for g in target_graphs]

        with mp.Pool(processes=self.num_workers) as pool:
            documents = pool.map(_process_single_graph_hybrid, tasks)

        return documents

    def create_vocab(self, documents):
        global_counts = Counter()
        for doc in documents:
            global_counts.update(doc)

        frequent_subgraphs = {word: count for word, count in global_counts.items() if count >= self.min_count}
        sorted_vocab = sorted(frequent_subgraphs.items(), key=lambda item: item[1], reverse=True)
        trimmed_vocab = sorted_vocab[:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, documents):
        sparse_vector = np.zeros([len(documents), self.n_vocab])
        for i, doc in enumerate(documents):
            words_count = Counter(doc)
            for j, (subgraph_sig, _) in enumerate(self.vocab):
                sparse_vector[i][j] = words_count[subgraph_sig]
        return sparse_vector

    def generate_training_embeddings(self, graphs):
        """Extracts subgraphs, builds vocabulary, filters collinearity, and returns Y matrix."""
        # 1. Parallel Extraction & Raw Vocab
        documents = self.extract_subgraphs_parallel(graphs)
        self.vocab = self.create_vocab(documents)
        raw_embeddings = self.calc_coefficients(documents)

        # 2. Pandas Redundancy Filter
        print(f"Filtering Collinear Subgraphs...")
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
        """Extracts subgraphs from new data and maps them to the EXISTING filtered vocabulary."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call generate_training_embeddings first.")

        documents = self.extract_subgraphs_parallel(graphs)
        return self.calc_coefficients(documents)