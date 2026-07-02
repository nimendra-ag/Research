"""
Running gSpan on the NCI dataset with edge labels.

Edge label strategy:
  Option A (recommended): bond_type alone -> matches NCI literature (|LE|=3)
  Option B: composite label from multiple bond attributes
"""

import os
import numpy as np
import networkx as nx
from rdkit import Chem
from graph_encoders.gspan import GSpan

# ===================================================================
# 1. Load NCI dataset (your code, unchanged)
# ===================================================================
print('Loading NCI dataset')
DATASET_DIR = "datasets/NCI_full"  # change this
id = 1  # dataset ID

graphs = []
y = []

filename = f"{id}total-connect.sdf"
filepath = os.path.join(DATASET_DIR, filename)

supplier = Chem.SDMolSupplier(filepath, sanitize=False, removeHs=False)
for mol in supplier:
    if mol is None:
        continue

    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            feature=atom.GetSymbol()
        )

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=str(bond.GetBondType()),
            bond_order=bond.GetBondTypeAsDouble(),
            aromatic=bond.GetIsAromatic(),
            in_ring=bond.IsInRing(),
            conjugated=bond.GetIsConjugated(),
            stereo=str(bond.GetStereo())
        )

    label = int(float(mol.GetProp("value")))
    graphs.append(G)
    y.append(label)

print(f"Loaded {len(graphs)} graphs")

# ===================================================================
# 2. Inspect edge label distribution (do this first)
# ===================================================================
from collections import Counter

bond_type_counts = Counter()
for G in graphs:
    for u, v, data in G.edges(data=True):
        bond_type_counts[data['bond_type']] += 1

print(f"\nBond type distribution:")
for bt, count in bond_type_counts.most_common():
    print(f"  {bt}: {count}")
print(f"  Unique bond types: {len(bond_type_counts)}")

# ===================================================================
# 3. Run gSpan
# ===================================================================

# --- Option A: bond_type as edge label (recommended) ---
# This matches the NCI literature: |LE| = 3 (SINGLE, DOUBLE, AROMATIC)
# and is what Thoma et al. (CORK, SDM 2009) used.

miner = GSpan(
    min_support=200,       # ~5% of dataset; tune as needed
    max_num_vertices=7,    # limit pattern size for tractability
    verbose=True
)
miner.run(
    graphs,
    node_label_attr='feature',
    edge_label_attr='bond_type',  # single attribute -> clean label space
)

# --- Option B: composite edge label (richer but slower) ---
# Uncomment below if you want to encode multiple attributes.
# WARNING: this creates many distinct labels, which slows mining
# significantly and fragments support. Only use if you have a
# specific reason to distinguish e.g. aromatic-in-ring from
# aromatic-not-in-ring.
#
# for G in graphs:
#     for u, v, data in G.edges(data=True):
#         G[u][v]['composite_label'] = (
#             f"{data['bond_type']}"
#             f"_ar{int(data['aromatic'])}"
#             f"_ri{int(data['in_ring'])}"
#         )
#
# miner = GSpan(min_support=200, max_num_vertices=7, verbose=True)
# miner.run(graphs, node_label_attr='feature',
#           edge_label_attr='composite_label')

# ===================================================================
# 4. Inspect results
# ===================================================================
results = miner.get_frequent_subgraphs_as_nx()

print(f"\nTotal frequent subgraphs found: {len(results)}")
print(f"  Vertex labels mapped: {miner.vlabel_map}")
print(f"  Edge labels mapped:   {miner.elabel_map}")

size_dist = Counter(r['num_vertices'] for r in results)
print(f"\nPattern size distribution:")
for s in sorted(size_dist):
    print(f"  |V|={s}: {size_dist[s]} patterns")

print("\n--- Sample Frequent Subgraphs ---")
for i, r in enumerate(results[:15]):
    nodes = {n: d['feature'] for n, d in r['graph'].nodes(data=True)}
    edges = {(u, v): d['label'] for u, v, d in r['graph'].edges(data=True)}
    print(f"  #{i+1}: |V|={r['num_vertices']}, |E|={r['num_edges']}, "
          f"support={r['support']}")
    print(f"         nodes={nodes}")
    print(f"         edges={edges}")
    print(f"         DFS code: {r['dfs_code']}")

# ===================================================================
# 5. Build binary indicator matrix (Def 2.3, CORK paper)
# ===================================================================
n_graphs = len(graphs)
n_features = len(results)

X = np.zeros((n_graphs, n_features), dtype=np.int8)
for feat_idx, r in enumerate(results):
    for gid in r['graph_ids']:
        X[gid, feat_idx] = 1

y_arr = np.array(y)

print(f"\nBinary indicator matrix: {X.shape}")
print(f"  Sparsity: {1 - X.mean():.4f}")
print(f"  Avg features per graph: {X.sum(axis=1).mean():.1f}")
print(f"  Class distribution: {dict(zip(*np.unique(y_arr, return_counts=True)))}")
