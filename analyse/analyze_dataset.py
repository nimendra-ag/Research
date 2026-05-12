import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import sys
import random

# Add the parent directory to the Python path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.graph_data import GraphDataLoader

def analyze_dataset(dataset_name="nci_full", dataset_id=1):
    """
    Analyzes the specified dataset and generates visualizations.
    """
    print(f"Starting analysis for dataset: {dataset_name} (ID: {dataset_id})")
    
    loader = GraphDataLoader()
    
    if dataset_name == "nci_full":
        graphs, labels = loader.load_nci_full(id=dataset_id)
    elif dataset_name == "nci_balanced":
        graphs, labels = loader.load_nci_balanced(id=dataset_id)
    elif dataset_name == "reddit10k":
        graphs = loader.reddit10k_graphs
        labels = loader.reddit10k_labels
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    if not graphs:
        print("No graphs loaded.")
        return

    # Create output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "analytics", f"{dataset_name}_{dataset_id}"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution
    print("Analyzing Class Level Insights...")
    label_counts = Counter(labels)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), hue=list(label_counts.keys()), palette="viridis", legend=False)
    plt.title("Class Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Graphs")
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()
    
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Dataset: {dataset_name} (ID: {dataset_id})\n")
        f.write(f"Total Graphs: {len(graphs)}\n")
        f.write("Class Distribution:\n")
        for label, count in label_counts.items():
            f.write(f"  Class {label}: {count} graphs ({count/len(graphs)*100:.2f}%)\n")
            
    # 2. Graph Size Distribution (Number of Nodes and Edges)
    print("Analyzing Graph Sizes...")
    num_nodes = [G.number_of_nodes() for G in graphs]
    num_edges = [G.number_of_edges() for G in graphs]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(num_nodes, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Node Count Distribution")
    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Frequency")
    
    sns.histplot(num_edges, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Edge Count Distribution")
    axes[1].set_xlabel("Number of Edges")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_sizes_distribution.png"))
    plt.close()

    # 3. Node Level Insights (Degree Distribution)
    print("Analyzing Node Level Insights (Degrees)...")
    avg_degrees = []
    all_degrees = []
    for G in graphs:
        degrees = [d for n, d in G.degree()]
        if degrees:
            avg_degrees.append(np.mean(degrees))
            all_degrees.extend(degrees)
            
    plt.figure(figsize=(8, 5))
    sns.histplot(all_degrees, kde=False, bins=range(max(all_degrees)+2), color="purple")
    plt.title("Overall Node Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "node_degree_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(avg_degrees, kde=True, color="green")
    plt.title("Average Degree per Graph Distribution")
    plt.xlabel("Average Degree")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "avg_degree_per_graph.png"))
    plt.close()

    # 4. Node Label Distribution
    print("Analyzing Node Labels...")
    all_labels = []
    for G in graphs:
        for _, data in G.nodes(data=True):
            if "label" in data:
                all_labels.append(data["label"])
                
    if all_labels:
        label_dist = Counter(all_labels)
        # Handle cases with many diverse labels
        top_labels = dict(label_dist.most_common(20))
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(top_labels.keys()), y=list(top_labels.values()), hue=list(top_labels.keys()), palette="magma", legend=False)
        plt.title("Top 20 Node Labels (Symbols) Distribution")
        plt.xlabel("Node Label")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "node_label_distribution.png"))
        plt.close()
    
    with open(os.path.join(output_dir, "summary.txt"), "a") as f:
        f.write(f"\nGraph Statistics:\n")
        f.write(f"  Average Nodes per Graph: {np.mean(num_nodes):.2f} (std: {np.std(num_nodes):.2f})\n")
        f.write(f"  Average Edges per Graph: {np.mean(num_edges):.2f} (std: {np.std(num_edges):.2f})\n")
        if avg_degrees:
            f.write(f"  Average Degree Overall: {np.mean(all_degrees):.2f}\n")
            
    # 5. Visualize Sample Graphs
    num_samples = min(5, len(graphs))
    print(f"Visualizing {num_samples} sample graphs...")
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    
    if num_samples == 1:
        axes = [axes]
        
    # Sample randomly to avoid getting graphs all from the same class at the start of the file
    sample_indices = random.sample(range(len(graphs)), num_samples)
        
    for idx, graph_idx in enumerate(sample_indices):
        G = graphs[graph_idx]
        ax = axes[idx]
        
        # Extract node labels if they exist (usually true for NCI datasets)
        node_labels = nx.get_node_attributes(G, 'label')
        
        # Use spring layout for a balanced graph drawing
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph structure
        nx.draw(G, pos, ax=ax, with_labels=False, node_color='lightblue', 
                node_size=300, edge_color='gray')
                
        # Draw node labels as atom symbols (for NCI) or just IDs
        if node_labels:
            nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=10, font_weight='bold')
        else:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            
        ax.set_title(f"Sample {idx+1}\nClass: {labels[graph_idx]}")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_graphs.png"), dpi=300)
    plt.close()

    print(f"Analysis successfully completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    # Feel free to change the dataset ID or name here 
    # Example NCI IDs: 1, 33, 41, 47, 81, 83, 109, 123, 145
    analyze_dataset("nci_balanced", dataset_id=1)
    
    # Or, to run reddit10k:
    # analyze_dataset("reddit10k", dataset_id=None)
