import os
import networkx as nx
from rdkit import Chem
from karateclub.dataset import GraphSetReader



def load_data(name: str, size: int):
    """
    Loads dataset
    nci -> NCI
    reddit -> reddit10k
    """

    if name == "nci":
        print('Loading NCI dataset')
        DATASET_DIR = "datasets/NCI_full"  # change this
        graphs = []
        y = []

        for filename in os.listdir(DATASET_DIR)[0:size]:
            if filename.endswith(".sdf"):
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

    elif name == "reddit":
        print('Loading reddit10k dataset')
        reader = GraphSetReader("reddit10k")

        graphs = reader.get_graphs()
        y = reader.get_target()
        print(f"Loaded {len(graphs)} graphs")
        return graphs, y

    print(f"Dataset {name} not found")
    return None

