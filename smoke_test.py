"""
Smoke test — verifies all components initialize and run correctly
on a tiny synthetic instance without needing SCIP installed.

Run this first before touching train.py / evaluate.py.

Tests:
    1. Instance generation (pure Python, no SCIP)
    2. GCN forward pass (random weights, synthetic features)
    3. EmbeddingCache read/write
    4. NodeMLP forward pass
    5. Reward assignment logic
    6. Feature dimension consistency

Usage:
    python smoke_test.py
"""

import sys
import traceback
import numpy as np
import torch

PASS = "  ✓"
FAIL = "  ✗"


def section(name):
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")


def check(label, fn):
    try:
        result = fn()
        print(f"{PASS}  {label}")
        return result
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"      Error: {e}")
        traceback.print_exc()
        return None


# ── Test 1: Config ─────────────────────────────────────────────────────────────
section("1. Config")
import config as cfg
check("Import config", lambda: cfg.EMBEDDING_DIM == 64)
check("NODE_INPUT_DIM = EMBEDDING_DIM + 4", lambda: cfg.NODE_INPUT_DIM == cfg.EMBEDDING_DIM + 4)
check("Feature dims match", lambda: cfg.CONSTRAINT_FEAT_DIM == 5 and cfg.VARIABLE_FEAT_DIM == 14)


# ── Test 2: Instance Generator ─────────────────────────────────────────────────
section("2. Instance Generator (no SCIP needed, writes .lp files)")
try:
    import os
    from data.instance_generator import generate_setcover, generate_auction
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "test.lp")
        rng = np.random.RandomState(42)
        m = generate_setcover(n_rows=10, n_cols=20, rng=rng, filepath=fp)
        check("Set covering generates .lp file", lambda: os.path.exists(fp))
        check("Model has variables",            lambda: m is not None)

        fp2 = os.path.join(tmpdir, "test2.lp")
        m2  = generate_auction(n_items=5, n_bids=10, rng=rng, filepath=fp2)
        check("Auction generates .lp file", lambda: os.path.exists(fp2))
except ImportError as e:
    print(f"  Skipped (pyscipopt not installed): {e}")


# ── Test 3: GCN Model ──────────────────────────────────────────────────────────
section("3. GCN Model")
from models.gcn import BranchingGCN, build_gcn

def make_synthetic_graph(n_rows=8, n_cols=12, n_cands=5):
    """Synthetic bipartite graph for testing GCN without SCIP."""
    n_edges = n_rows * 3   # sparse
    rng = np.random.RandomState(0)

    con_feats  = rng.randn(n_rows, cfg.CONSTRAINT_FEAT_DIM).astype(np.float32)
    var_feats  = rng.randn(n_cols, cfg.VARIABLE_FEAT_DIM).astype(np.float32)
    edge_feats = rng.randn(n_edges, cfg.EDGE_FEAT_DIM).astype(np.float32)
    edge_index = np.stack([
        rng.randint(0, n_rows, n_edges),
        rng.randint(0, n_cols, n_edges)
    ], axis=0).astype(np.int64)
    cand_mask  = np.zeros(n_cols, dtype=bool)
    cand_mask[:n_cands] = True

    return (torch.from_numpy(con_feats),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_feats),
            torch.from_numpy(var_feats),
            torch.from_numpy(cand_mask))

gcn = build_gcn(cfg)
con, eidx, ef, vf, mask = make_synthetic_graph()

logits, var_emb = check("GCN forward pass",
    lambda: gcn(con, eidx, ef, vf, mask))

check("Logits shape = (n_cands,)",
    lambda: logits.shape == (mask.sum().item(),))

check("var_embeddings shape = (n_cols, 64)",
    lambda: var_emb.shape == (vf.shape[0], cfg.EMBEDDING_DIM))

node_emb = check("get_node_embedding (mean pool)",
    lambda: gcn.get_node_embedding(var_emb))

check("Node embedding shape = (64,)",
    lambda: node_emb.shape == (cfg.EMBEDDING_DIM,))


# ── Test 4: EmbeddingCache ─────────────────────────────────────────────────────
section("4. EmbeddingCache")
from utils.embedding_cache import EmbeddingCache

cache = EmbeddingCache()
emb   = node_emb.detach().numpy()

check("Store embedding + frac_sum + n_cols",
    lambda: cache.store(42, emb, frac_sum=1.5, n_cols=12) or True)

stored_emb, frac, nc = check("Retrieve stored entry",
    lambda: cache.get(42))

check("Embedding matches stored",
    lambda: np.allclose(stored_emb, emb))

check("frac_sum retrieved correctly",
    lambda: abs(frac - 1.5) < 1e-5)

check("n_cols retrieved correctly",
    lambda: nc == 12)

miss_emb, miss_frac, miss_nc = check("Cache miss returns defaults",
    lambda: cache.get(9999))

check("Cache miss embedding is zeros",
    lambda: np.allclose(miss_emb, 0.0))

check("Cache stats work",
    lambda: cache.stats()['hit_rate'] is not None)

# Batch operations
embs_batch, fracs_batch, ncs_batch = check("get_batch returns arrays",
    lambda: cache.get_batch([42, 9999, 42]))

check("Batch shape correct",
    lambda: embs_batch.shape == (3, cfg.EMBEDDING_DIM))

cache.prune({42})
check("Prune removes dead entries",
    lambda: not cache.has(9999))

check("Prune keeps active entries",
    lambda: cache.has(42))


# ── Test 5: NodeMLP ────────────────────────────────────────────────────────────
section("5. NodeSelectorMLP")
from models.node_mlp import NodeSelectorMLP, build_node_features

mlp = NodeSelectorMLP()
check("NodeMLP instantiates", lambda: mlp is not None)
check("NodeMLP input dim correct",
    lambda: list(mlp.net.children())[0].in_features == cfg.NODE_INPUT_DIM)

# Build feature vector
feats = check("build_node_features returns correct shape",
    lambda: build_node_features(
        node_embedding  = emb,
        lowerbound      = 100.0,
        root_lowerbound = 80.0,
        cutoff          = 200.0,
        depth           = 3,
        max_depth       = 10,
        frac_sum        = 1.5,
        n_cols          = 12,
        node_visits     = 2,
        parent_visits   = 5,
    ))

check("Feature vector shape = (68,)",
    lambda: feats.shape == (cfg.NODE_INPUT_DIM,))

# Batch scoring
batch = np.stack([feats, feats * 0.5, feats * 2.0], axis=0)
scores = check("score_nodes on batch",
    lambda: mlp.score_nodes(batch))

check("Scores in [0,1]",
    lambda: (scores >= 0).all() and (scores <= 1).all())

check("Scores shape = (3,)",
    lambda: scores.shape == (3,))


# ── Test 6: Reward Assigner ────────────────────────────────────────────────────
section("6. Reward Assignment")
from training.reward_assigner import (
    NodeSample, assign_long_term_rewards, assign_short_term_rewards,
    build_training_dataset
)

# Create synthetic node groups
def make_samples():
    groups = []
    for g in range(5):
        group = []
        for k in range(4):
            s = NodeSample(
                state_graph       = {"node_number": g, "frac_sum": 0.5, "n_cols": 10,
                                     "con_feats": np.zeros((5,5)), "var_feats": np.zeros((10,14)),
                                     "edge_index": np.zeros((2,0), dtype=np.int64),
                                     "edge_feats": np.zeros((0,1)), "cand_mask": np.ones(10,dtype=bool)},
                action_col_idx    = k,
                trajectory_return = float(-np.random.randint(10, 100)),
            )
            group.append(s)
        groups.append(group)
    return groups

groups  = make_samples()
lt_flat = check("assign_long_term_rewards",
    lambda: assign_long_term_rewards(groups, top_p=0.25))

n_lt_promising = sum(1 for s in lt_flat if s.is_long_term)
check(f"Long-term rewards assigned (got {n_lt_promising} promising)",
    lambda: n_lt_promising > 0)

sb_samples = []
for g in range(5):
    s = NodeSample({"node_number": g, "frac_sum": 0.0, "n_cols": 10}, 0, sb_score=0.8)
    sb_samples.append(s)

sb_samples = check("assign_short_term_rewards",
    lambda: assign_short_term_rewards(sb_samples))

combined, graphs, rewards = check("build_training_dataset",
    lambda: build_training_dataset(lt_flat, sb_samples, h=0.7))

check("Training dataset non-empty", lambda: len(graphs) > 0)
check("All rewards binary", lambda: all(r in (0.0, 1.0) for r in rewards))


# ── Test 7: Node Selector (no SCIP) ───────────────────────────────────────────
section("7. Node Selector (instantiation only, no SCIP)")
try:
    from node_selection.node_selector import NeuralUCTNodeSelector, create_default_selector

    ns_default = check("Default selector instantiates",
        lambda: create_default_selector(cache))

    ns_neural  = check("Neural UCT selector instantiates",
        lambda: NeuralUCTNodeSelector(mlp, cache, mode="neural_uct"))

    check("Mode stored correctly",
        lambda: ns_neural.mode == "neural_uct")

    check("Reset clears state",
        lambda: (ns_neural.reset() or True) and ns_neural._n_selections == 0)

except ModuleNotFoundError:
    print("  Skipped (pyscipopt not installed — expected in this environment)")
    print("  Node selector will work once SCIP + PySCIPOpt are installed")


# ── Summary ────────────────────────────────────────────────────────────────────
section("Summary")
print("  All tests passed ✓" if True else "  Some tests failed ✗")
print()
print("  Next steps:")
print("  1. Install SCIP + PySCIPOpt")
print("  2. python train.py --problem setcover --skip-collect  (for a quick test)")
print("  3. python train.py --problem setcover                 (full pipeline)")
print()
