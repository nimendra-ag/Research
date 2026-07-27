"""
Evaluate karateclub's Graph2Vec on the NCI1 dataset.

Follows the paper's §5.2 protocol (real-world datasets):
  - 70/30 stratified train/test split
  - SVM classifier with hyperparameters tuned via 5-fold CV on train set
  - Repeated 5 times; report mean ± std accuracy
  - Embedding dimension = 1024
"""

import argparse
import inspect
import logging
import sys
import time

import networkx as nx
import numpy as np
from karateclub import Graph2Vec
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, ".")
from utils.graph_data import GraphDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_graph2vec")


def _supports_node_attribute() -> bool:
    """Check if installed karateclub version supports use_node_attribute."""
    sig = inspect.signature(Graph2Vec.__init__)
    return "use_node_attribute" in sig.parameters


def _encode_node_features(graphs: list[nx.Graph], attr: str = "feature") -> list[nx.Graph]:
    """
    Build a global string→int mapping for node features and write them
    back as integer 'label' attributes.

    This is the fallback for older karateclub versions that lack
    `use_node_attribute`: by relabeling nodes with feature-derived
    integers, the default WL hashing (which uses node degree) is
    bypassed via an explicit 'label' attribute.
    """
    label_map: dict[str, int] = {}
    processed = []

    for g in graphs:
        g_copy = g.copy()
        for node in g_copy.nodes():
            feat = str(g_copy.nodes[node].get(attr, g_copy.degree(node)))
            if feat not in label_map:
                label_map[feat] = len(label_map)
            g_copy.nodes[node]["label"] = label_map[feat]
        processed.append(g_copy)

    logger.info("Encoded %d distinct node features into integer labels", len(label_map))
    return processed


def evaluate_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_repeats: int = 5,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    SVM evaluation following the paper's protocol.

    - 70/30 stratified split (§5.2 real-world setup)
    - RBF SVM, C tuned via 5-fold CV on train set
    - Repeat n_repeats times, report mean ± std for every metric

    Returns
    -------
    dict mapping metric name → (mean, std) as percentages.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=n_repeats, test_size=test_size, random_state=seed
    )
    param_grid = {"svc__C": [0.01, 0.1, 1, 10, 100]}

    fold_metrics: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
    }

    for fold, (train_idx, test_idx) in enumerate(splitter.split(embeddings, labels), 1):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # probability=True enables predict_proba for AUC computation
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True)),
        ])
        cv = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        cv.fit(X_train, y_train)

        best_model = cv.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)
        fold_metrics["auc"].append(auc)

        logger.info(
            "Fold %d/%d — acc: %.2f%%  prec: %.2f%%  rec: %.2f%%  "
            "f1: %.2f%%  auc: %.4f  (C=%.2f)",
            fold, n_repeats,
            acc * 100, prec * 100, rec * 100, f1 * 100, auc,
            cv.best_params_["svc__C"],
        )

    results = {
        name: (float(np.mean(vals) * 100), float(np.std(vals) * 100))
        for name, vals in fold_metrics.items()
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="graph2vec on NCI1")
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimension (δ)")
    parser.add_argument("--wl-depth", type=int, default=3, help="WL iteration depth (D)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (ε)")
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate (α)")
    parser.add_argument("--min-count", type=int, default=5, help="Min subgraph frequency")
    parser.add_argument("--workers", type=int, default=4, help="Gensim worker threads")
    parser.add_argument("--n-repeats", type=int, default=5, help="Eval repeats")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # ---- Load NCI1 -------------------------------------------------------
    logger.info("Loading NCI1 dataset…")
    loader = GraphDataLoader()
    graphs, labels = loader.nci_full_graphs, loader.nci_full_labels
    labels = np.array(labels)
    logger.info("Loaded %d graphs, %d classes", len(graphs), len(set(labels)))

    # ---- Handle node features across karateclub versions -----------------
    use_attr = _supports_node_attribute()

    if use_attr:
        logger.info("karateclub supports use_node_attribute — using 'feature' directly")
        model = Graph2Vec(
            wl_iterations=args.wl_depth,
            use_node_attribute="feature",
            dimensions=args.dim,
            epochs=args.epochs,
            learning_rate=args.lr,
            min_count=args.min_count,
            workers=args.workers,
            seed=args.seed,
        )
    else:
        logger.info(
            "karateclub lacks use_node_attribute — encoding features into 'label' attr"
        )
        graphs = _encode_node_features(graphs, attr="feature")
        model = Graph2Vec(
            wl_iterations=args.wl_depth,
            attributed=True,          # older API flag
            dimensions=args.dim,
            epochs=args.epochs,
            learning_rate=args.lr,
            min_count=args.min_count,
            workers=args.workers,
            seed=args.seed,
        )

    # ---- Train graph2vec -------------------------------------------------
    logger.info(
        "Training graph2vec — dim=%d, wl_depth=%d, epochs=%d",
        args.dim, args.wl_depth, args.epochs,
    )
    t0 = time.perf_counter()
    model.fit(graphs)
    pre_training_time = time.perf_counter() - t0
    logger.info("Pre-training completed in %.1f s", pre_training_time)

    embeddings = model.get_embedding()
    logger.info("Embedding shape: %s", embeddings.shape)

    # ---- Evaluate --------------------------------------------------------
    logger.info("Evaluating with SVM (5-fold CV for C tuning, 70/30 split)…")
    results = evaluate_embeddings(
        embeddings, labels, n_repeats=args.n_repeats, seed=args.seed,
    )

    logger.info("=" * 55)
    logger.info("NCI1 — graph2vec results (%d repeats)", args.n_repeats)
    logger.info("-" * 55)
    for metric, (mean, std) in results.items():
        logger.info("  %-12s: %.2f ± %.2f%%", metric, mean, std)
    logger.info("-" * 55)
    logger.info("Paper accuracy ref:  73.22 ± 1.81%%")
    logger.info("Pre-training time:   %.1f s", pre_training_time)
    logger.info("=" * 55)


if __name__ == "__main__":
    main()