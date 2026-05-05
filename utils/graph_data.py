import os
import networkx as nx
from rdkit import Chem
from karateclub.dataset import GraphSetReader

class GraphDataLoader:

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._nci_full_graphs = None
            self._nci_full_labels = None
            self._nci_balanced_graphs = None
            self._nci_balanced_labels = None
            self._reddit10k_graphs = None
            self._reddit10k_labels = None
            self._initialized = True

    @property
    def nci_full_graphs(self):
        if self._nci_full_graphs is None:
            self._nci_full_graphs, self._nci_full_labels = self.load_nci_full()
        return self._nci_full_graphs

    @property
    def nci_full_labels(self):
        if self._nci_full_labels is None:
            self._nci_full_graphs, self._nci_full_labels = self.load_nci_full()
        return self._nci_full_labels
    
    @property
    def nci_balanced_graphs(self):
        if self._nci_balanced_graphs is None:
            self._nci_balanced_graphs, self._nci_balanced_labels = self.load_nci_balanced()
        return self._nci_balanced_graphs

    @property
    def nci_balanced_labels(self):
        if self._nci_balanced_labels is None:
            self._nci_balanced_graphs, self._nci_balanced_labels = self.load_nci_balanced()
        return self._nci_balanced_labels

    @property
    def reddit10k_graphs(self):
        if self._reddit10k_graphs is None:
            self._reddit10k_graphs, self._reddit10k_labels = self.load_reddit10k()
        return self._reddit10k_graphs
    
    @property
    def reddit10k_labels(self):
        if self._reddit10k_labels is None:
            self._reddit10k_graphs, self._reddit10k_labels = self.load_reddit10k()
        return self._reddit10k_labels

    def load_nci_full(self, id=1):
        """
        id - (1, 33, 41, 47, 81, 83, 109, 123, 145)
        """
        print('Loading NCI dataset')
        DATASET_DIR = "datasets/NCI_full"  # change this
        graphs = []
        y = []

        filename = f"{id}total-connect.sdf"
        filepath = os.path.join(DATASET_DIR, filename)

        supplier = Chem.SDMolSupplier(filepath, sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue

            G = nx.Graph()

            # Add atoms as nodes
            for atom in mol.GetAtoms():
                G.add_node(
                    atom.GetIdx(),
                    label=atom.GetSymbol()   # WL uses node labels
                )

            # Add bonds as edges
            for bond in mol.GetBonds():
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx()
                )

            # Get graph label
            # In NCI, class label is stored as a molecule property
            label = int(float(mol.GetProp("value")))
            graphs.append(G)
            y.append(label)

        print(f"Loaded {len(graphs)} graphs")
        return graphs, y

    def load_nci_balanced(self, id=1):
        """
        id - (1, 33, 41, 47, 81, 83, 109, 123, 145)
        """
        print('Loading NCI balanced dataset')
        DATASET_DIR = "datasets/NCI_balanced"
        graphs = []
        y = []

        filename = f"{id}-balance.sdf"
        filepath = os.path.join(DATASET_DIR, filename)

        supplier = Chem.SDMolSupplier(filepath, sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue

            G = nx.Graph()

            # Add atoms as nodes
            for atom in mol.GetAtoms():
                G.add_node(
                    atom.GetIdx(),
                    label=atom.GetSymbol()   # WL uses node labels
                )

            # Add bonds as edges
            for bond in mol.GetBonds():
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx()
                )

            # Get graph label
            label = int(float(mol.GetProp("value")))
            graphs.append(G)
            y.append(label)

        print(f"Loaded {len(graphs)} graphs from balanced dataset")
        return graphs, y

    def load_reddit10k(self):
        print('Loading reddit10k dataset')
        reader = GraphSetReader("reddit10k")

        graphs = reader.get_graphs()
        y = reader.get_target()
        print(f"Loaded {len(graphs)} graphs")
        return graphs, y