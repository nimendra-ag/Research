"""
visualize_wl.py
================
A standalone, self-contained script that walks through the ImbalanceAwareWL
feature pipeline (graph_encoders/wl.py) and *visualizes every intermediate
result* so you can literally see how the score calculation and sorting work.

It does NOT touch anything under implements/. It only:
  1. Loads the NCI graph dataset (via utils/graph_data.py)
  2. Runs Weisfeiler-Lehman hashing to turn each graph into a bag of "words"
  3. Re-derives the exact scoring math from WL.create_vocab (document
     frequencies -> per-class probabilities -> sqrt discriminative score ->
     total-presence weighting -> sort -> 50% threshold selection)
  4. Builds the final sparse feature/count matrix (WL.calc_coefficients)
  5. Saves a series of annotated plots to  viz_output/<timestamp>/

Run it, then open the PNGs in that folder. See the bottom of the file / the
chat message for the exact run command.
"""

import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save to files, no blocking window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# make the repo importable regardless of where we launch from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.graph_data import GraphDataLoader          # noqa: E402
from graph_encoders.wl import WL                        # noqa: E402
from sklearn.model_selection import train_test_split    # noqa: E402
from sklearn.decomposition import PCA                    # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SUBSAMPLE = 6000        # stratified subsample for speed; set None to use all
NCI_ID = 1              # which NCI file to load (1,33,41,47,81,83,109,123,145)
TOP_N_WORDS = 20        # how many top discriminative words to chart
SEED = 42

# ---- palette (colorblind-safe blue/orange categorical pair) --------------- #
C_MAJ = "#4C72B0"       # majority  (label == -1)
C_MIN = "#DD8452"       # minority  (label != -1)
C_SEL = "#4C72B0"       # selected / kept
C_DROP = "#BFBFBF"      # dropped / trimmed
C_ACCENT = "#C44E52"    # threshold / annotation lines
SEQ_CMAP = "viridis"    # sequential ramp for magnitude (score)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "viz_output",
    datetime.now().strftime("%Y%m%d_%H%M%S"),
)
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {name}")


# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #
print("Loading dataset ...")
loader = GraphDataLoader()
graphs, y = loader.nci_full_graphs, loader.nci_full_labels
y = np.array(y, dtype=float)
print(f"Full dataset: {len(graphs)} graphs "
      f"(majority={int((y == -1).sum())}, minority={int((y != -1).sum())})")

# stratified subsample so the run is fast but keeps the class imbalance
if SUBSAMPLE and SUBSAMPLE < len(graphs):
    graphs, _, y, _ = train_test_split(
        graphs, y, train_size=SUBSAMPLE, stratify=y, random_state=SEED
    )
    print(f"Subsampled to {len(graphs)} graphs (stratified).")

n_maj = int((y == -1).sum())
n_min = int((y != -1).sum())

# --------------------------------------------------------------------------- #
# 2. WL hashing: graph -> bag of subtree-hash "words"
# --------------------------------------------------------------------------- #
print("Running WL hashing ...")
wl = WL()
wl._set_seed()
documents = wl.create_wl_hash(graphs)   # list of TaggedDocument, .words = hashes

# --------------------------------------------------------------------------- #
# 3. Re-derive the scoring math EXACTLY as WL.create_vocab does, but keep every
#    intermediate array so we can plot it.
# --------------------------------------------------------------------------- #
print("Computing discriminative scores ...")
majority_df, minority_df = Counter(), Counter()
maj_graphs = min_graphs = 0

for doc, label in zip(documents, y):
    unique_words = Counter(doc.words)        # DF: count a word once per graph
    if label == -1:
        maj_graphs += 1
        for w in unique_words:
            majority_df[w] += 1
    else:
        min_graphs += 1
        for w in unique_words:
            minority_df[w] += 1

all_words = list(set(majority_df) | set(minority_df))

words, p_maj_a, p_min_a, disc_a, total_a, score_a, mdf_a, ndf_a = ([] for _ in range(8))
for w in all_words:
    p_majority = majority_df[w] / maj_graphs
    p_minority = minority_df[w] / min_graphs
    discriminative = abs(np.sqrt(p_majority) - np.sqrt(p_minority))
    total_presence = p_majority + p_minority
    score = total_presence * discriminative

    words.append(w)
    p_maj_a.append(p_majority)
    p_min_a.append(p_minority)
    disc_a.append(discriminative)
    total_a.append(total_presence)
    score_a.append(score)
    mdf_a.append(majority_df[w])
    ndf_a.append(minority_df[w])

p_maj_a = np.array(p_maj_a); p_min_a = np.array(p_min_a)
disc_a = np.array(disc_a);   total_a = np.array(total_a)
score_a = np.array(score_a); mdf_a = np.array(mdf_a); ndf_a = np.array(ndf_a)

# sort by score descending (this is the "sorting process")
order = np.argsort(-score_a)
scores_sorted = score_a[order]

# 50% decile threshold, exactly like create_vocab
l_scores = len(scores_sorted)
temp = int(l_scores * 0.50)
threshold = scores_sorted[temp]
selected_mask = score_a >= threshold
n_selected = int(selected_mask.sum())

print(f"Total features (words): {l_scores}")
print(f"Threshold (50% decile): {threshold:.6f}")
print(f"Selected {n_selected} / {l_scores} words")

# --------------------------------------------------------------------------- #
# 4. Build the final feature matrix via the real WL method (cross-check)
# --------------------------------------------------------------------------- #
print("Building feature/count matrix ...")
# reproduce WL's selection to set wl.vocab & wl.n_vocab, then count
scored_vocab = sorted(zip(words, score_a), key=lambda x: x[1], reverse=True)
trimmed_vocab = [it for it in scored_vocab if it[1] >= threshold]
if len(trimmed_vocab) < 50:
    trimmed_vocab = scored_vocab[: wl.n_vocab]
wl.vocab = trimmed_vocab
wl.n_vocab = len(trimmed_vocab)
embeddings = wl.calc_coefficients(documents)   # (n_graphs, n_vocab) counts
print(f"Feature matrix shape: {embeddings.shape}")


# =========================================================================== #
#  VISUALIZATIONS
# =========================================================================== #
print(f"\nWriting plots to: {OUT_DIR}\n")

# --- Fig 1: class imbalance ------------------------------------------------ #
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["Majority\n(label = -1)", "Minority\n(label = +1)"],
              [n_maj, n_min], color=[C_MAJ, C_MIN], width=0.6)
for b, v in zip(bars, [n_maj, n_min]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v/(n_maj+n_min)*100:.1f}%)",
            ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("number of graphs")
ax.set_title("Step 0 — Class imbalance drives the scoring")
ax.set_ylim(0, max(n_maj, n_min) * 1.18)
_save(fig, "01_class_imbalance.png")

# --- Fig 2: what one graph becomes (a WL 'document') ----------------------- #
sample_words = documents[0].words
top_sample = Counter(sample_words).most_common(12)
fig, ax = plt.subplots(figsize=(7, 4))
labels = [w[:8] + "…" for w, _ in top_sample]
ax.barh(range(len(top_sample)), [c for _, c in top_sample], color=C_MAJ)
ax.set_yticks(range(len(top_sample)))
ax.set_yticklabels(labels, fontfamily="monospace", fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("count within this single graph")
ax.set_title(f"Step 1 — One graph -> bag of WL 'words'\n"
             f"(graph #0 has {len(sample_words)} tokens, "
             f"{len(set(sample_words))} unique)")
_save(fig, "02_single_graph_words.png")

# --- Fig 3: document frequency scatter ------------------------------------- #
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(mdf_a, ndf_a, c=score_a, cmap=SEQ_CMAP, s=14,
                alpha=0.7, edgecolors="none")
ax.set_xlabel("majority document frequency  (# majority graphs with word)")
ax.set_ylabel("minority document frequency")
ax.set_title("Step 2 — Document frequency per class\n(color = final score)")
plt.colorbar(sc, ax=ax, label="score")
_save(fig, "03_document_frequency.png")

# --- Fig 4: per-class probabilities + the sqrt transform ------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
sc = axes[0].scatter(p_maj_a, p_min_a, c=score_a, cmap=SEQ_CMAP, s=14,
                     alpha=0.75, edgecolors="none")
axes[0].plot([0, 1], [0, 1], color=C_DROP, ls="--", lw=1,
             label="equally present (score→0)")
axes[0].set_xlabel(r"$p_{maj}$  = maj_df / #maj graphs")
axes[0].set_ylabel(r"$p_{min}$  = min_df / #min graphs")
axes[0].set_title("Step 3a — Per-class presence probability")
axes[0].legend(loc="upper left", fontsize=8, frameon=False)
plt.colorbar(sc, ax=axes[0], label="score")

xs = np.linspace(0, 1, 200)
axes[1].plot(xs, xs, color=C_DROP, ls="--", lw=1.2, label="identity")
axes[1].plot(xs, np.sqrt(xs), color=C_ACCENT, lw=2, label=r"$\sqrt{p}$")
axes[1].annotate("stretches small\nprobabilities", xy=(0.08, np.sqrt(0.08)),
                 xytext=(0.28, 0.55), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT))
axes[1].set_xlabel("p")
axes[1].set_ylabel("transformed")
axes[1].set_title("Step 3b — Why sqrt: it amplifies rare-word gaps")
axes[1].legend(loc="lower right", fontsize=9, frameon=False)
_save(fig, "04_probabilities_and_sqrt.png")

# --- Fig 5: score = total_presence * discriminative ------------------------ #
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(disc_a, total_a, c=score_a, cmap=SEQ_CMAP, s=16,
                alpha=0.8, edgecolors="none")
ax.set_xlabel(r"discriminative $= |\sqrt{p_{maj}} - \sqrt{p_{min}}|$")
ax.set_ylabel(r"total presence $= p_{maj} + p_{min}$")
ax.set_title("Step 4 — Final score = total_presence × discriminative\n"
             "(top-right = strong & discriminative = high score)")
plt.colorbar(sc, ax=ax, label="score")
_save(fig, "05_score_composition.png")

# --- Fig 6: score histogram with 50% threshold ---------------------------- #
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(score_a[score_a >= threshold], bins=60, color=C_SEL,
        alpha=0.9, label=f"kept ({n_selected})")
ax.hist(score_a[score_a < threshold], bins=60, color=C_DROP,
        alpha=0.9, label=f"dropped ({l_scores - n_selected})")
ax.axvline(threshold, color=C_ACCENT, lw=2,
           label=f"threshold = {threshold:.4f}")
ax.set_xlabel("score")
ax.set_ylabel("number of words")
ax.set_title("Step 5 — Score distribution & 50%-decile threshold")
ax.legend(frameon=False)
_save(fig, "06_score_histogram.png")

# --- Fig 7: the SORTING — rank vs score, selected region shaded ------------ #
fig, ax = plt.subplots(figsize=(8, 4.5))
ranks = np.arange(1, l_scores + 1)
ax.plot(ranks, scores_sorted, color=C_MAJ, lw=1.6)
ax.axhline(threshold, color=C_ACCENT, ls="--", lw=1.2)
ax.axvline(temp, color=C_ACCENT, ls="--", lw=1.2)
ax.fill_between(ranks, scores_sorted, where=(ranks <= temp),
                color=C_SEL, alpha=0.25, label="kept (top ~50%)")
ax.fill_between(ranks, scores_sorted, where=(ranks > temp),
                color=C_DROP, alpha=0.4, label="dropped")
ax.annotate(f"cut at rank {temp}\nscore = {threshold:.4f}",
            xy=(temp, threshold), xytext=(temp * 1.15, threshold + scores_sorted.max() * 0.2),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C_ACCENT))
ax.set_xlabel("rank (words sorted by score, descending)")
ax.set_ylabel("score")
ax.set_title("Step 6 — Sorting: words ranked high→low, top 50% kept")
ax.legend(frameon=False)
_save(fig, "07_sorted_score_curve.png")

# --- Fig 8: top-N discriminative words, p_maj vs p_min --------------------- #
top_idx = order[:TOP_N_WORDS]
yy = np.arange(TOP_N_WORDS)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(yy - 0.2, p_maj_a[top_idx], height=0.4, color=C_MAJ, label=r"$p_{maj}$")
ax.barh(yy + 0.2, p_min_a[top_idx], height=0.4, color=C_MIN, label=r"$p_{min}$")
for i, idx in enumerate(top_idx):
    ax.text(max(p_maj_a[idx], p_min_a[idx]) + 0.01, i,
            f"score={score_a[idx]:.3f}", va="center", fontsize=7, color="#444")
ax.set_yticks(yy)
ax.set_yticklabels([words[i][:8] + "…" for i in top_idx],
                   fontfamily="monospace", fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("presence probability")
ax.set_title(f"Step 7 — Top {TOP_N_WORDS} discriminative words\n"
             "(large gap between blue & orange = high score)")
ax.legend(frameon=False, loc="lower right")
_save(fig, "08_top_words.png")

# --- Fig 9: final feature matrix heatmap (rows grouped by class) ----------- #
maj_rows = np.where(y == -1)[0][:30]
min_rows = np.where(y != -1)[0][:30]
rows = np.concatenate([maj_rows, min_rows])
# embedding columns are the kept vocab, already sorted by score desc, so the
# top-scoring words are simply the first columns of the matrix.
n_cols = min(40, embeddings.shape[1])
cols = np.arange(n_cols)
col_words = [wl.vocab[j][0] for j in cols]
sub = embeddings[np.ix_(rows, cols)]
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(sub, aspect="auto", cmap="magma")
ax.axhline(len(maj_rows) - 0.5, color="white", lw=2)
ax.text(-0.6, len(maj_rows) / 2, "majority", rotation=90,
        va="center", ha="right", color=C_MAJ, fontweight="bold")
ax.text(-0.6, len(maj_rows) + len(min_rows) / 2, "minority", rotation=90,
        va="center", ha="right", color=C_MIN, fontweight="bold")
ax.set_xlabel("top discriminative words (vocab columns)")
ax.set_ylabel("graphs")
ax.set_title("Step 8 — Final feature matrix (word counts per graph)")
plt.colorbar(im, ax=ax, label="count")
_save(fig, "09_feature_matrix_heatmap.png")

# --- Fig 10: PCA of final embeddings, colored by class -------------------- #
print("Computing PCA projection ...")
pca = PCA(n_components=2, random_state=SEED)
proj = pca.fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(proj[y == -1, 0], proj[y == -1, 1], s=8, alpha=0.4,
           color=C_MAJ, label="majority", edgecolors="none")
ax.scatter(proj[y != -1, 0], proj[y != -1, 1], s=14, alpha=0.7,
           color=C_MIN, label="minority", edgecolors="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("Step 9 — Do the WL features separate the classes?\n"
             "PCA of the final feature matrix")
ax.legend(frameon=False)
_save(fig, "10_pca_embedding.png")

# --------------------------------------------------------------------------- #
# Text summary: a concrete numeric walkthrough of the #1 word
# --------------------------------------------------------------------------- #
best = order[0]
summary = f"""
=========================================================================
 WL SCORING — WORKED NUMERIC EXAMPLE (top-ranked word)
=========================================================================
 majority graphs : {maj_graphs}
 minority graphs : {min_graphs}
 total words     : {l_scores}

 Top word (hash): {words[best]}
   majority_df   = {mdf_a[best]}   -> p_maj = {mdf_a[best]}/{maj_graphs} = {p_maj_a[best]:.4f}
   minority_df   = {ndf_a[best]}   -> p_min = {ndf_a[best]}/{min_graphs} = {p_min_a[best]:.4f}
   discriminative= |sqrt({p_maj_a[best]:.4f}) - sqrt({p_min_a[best]:.4f})| = {disc_a[best]:.4f}
   total_presence= {p_maj_a[best]:.4f} + {p_min_a[best]:.4f} = {total_a[best]:.4f}
   SCORE         = {total_a[best]:.4f} * {disc_a[best]:.4f} = {score_a[best]:.4f}

 Sorting: {l_scores} words ranked by score (desc); cut at rank {temp}
          threshold score = {threshold:.4f}; kept {n_selected} words.
 Final feature matrix: {embeddings.shape[0]} graphs x {embeddings.shape[1]} words
=========================================================================
"""
print(summary)
with open(os.path.join(OUT_DIR, "00_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary)

print(f"Done. Open the PNGs in:\n  {OUT_DIR}")
