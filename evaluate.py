"""
Evaluate trained policies and run GO/NO GO comparison.

Compares:
    1. SCIP default (relpcost + hybridestim)
    2. Learned GCN branching + SCIP default node selection
    3. Learned GCN branching + Neural UCT node selection  ← your contribution

Usage:
    python evaluate.py [--problem setcover] [--difficulty easy]
    python evaluate.py --collect-node-data   # collect training data for NodeMLP
"""

import os
import argparse
import torch
import pickle

import config as cfg
from models.gcn import build_gcn
from models.node_mlp import NodeSelectorMLP
from utils.embedding_cache import EmbeddingCache
from utils.metrics import evaluate_policy, print_results, compare_policies
from branching.branch_rule import create_learned_branchrule, create_default_branchrule
from node_selection.node_selector import create_default_selector, create_neural_uct_selector
from data.feature_extractor import extract_bipartite_graph
from models.node_mlp import build_node_features
from training.reward_assigner import build_node_training_labels
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--problem",            default=cfg.PROBLEM_TYPE)
    p.add_argument("--difficulty",         default="easy")
    p.add_argument("--n-instances",        type=int, default=20)
    p.add_argument("--n-seeds",            type=int, default=cfg.EVAL_SEEDS)
    p.add_argument("--collect-node-data",  action="store_true")
    p.add_argument("--device",             default=cfg.DEVICE)
    return p.parse_args()


def load_gcn(problem_type, device):
    gcn_path = os.path.join(cfg.CHECKPOINT_DIR, f"{problem_type}_gcn_best.pt")
    gcn = build_gcn(cfg)
    if os.path.exists(gcn_path):
        ckpt = torch.load(gcn_path, map_location=device)
        gcn.load_state_dict(ckpt['model_state'])
        gcn.eval()
        print(f"Loaded GCN from {gcn_path}")
    else:
        print(f"WARNING: No GCN checkpoint at {gcn_path} — using random weights")
    return gcn


def load_node_mlp(problem_type, device):
    path = os.path.join(cfg.CHECKPOINT_DIR, f"{problem_type}_node_mlp.pt")
    mlp  = NodeSelectorMLP()
    if os.path.exists(path):
        mlp.load_state_dict(torch.load(path, map_location=device))
        mlp.eval()
        print(f"Loaded NodeMLP from {path}")
        return mlp
    else:
        print(f"WARNING: No NodeMLP checkpoint at {path} — using default node selection")
        return None


def get_instance_paths(problem_type, difficulty, n_instances):
    inst_dir = os.path.join(cfg.INSTANCE_DIR, problem_type, difficulty)
    if not os.path.exists(inst_dir):
        print(f"No instances found at {inst_dir}. Run train.py first.")
        return []
    paths = sorted([
        os.path.join(inst_dir, f)
        for f in os.listdir(inst_dir)
        if f.endswith('.lp')
    ])[:n_instances]
    print(f"Found {len(paths)} instances in {inst_dir}")
    return paths


class SolverBundle:
    """
    Holds a SHARED EmbeddingCache instance between the branching rule
    and node selector. This is the correct way to share embeddings.

    The bug this fixes: creating separate EmbeddingCache instances in two
    factory closures means the node selector never sees any embeddings
    (cache is always empty), defeating the whole Neural UCT approach.

    Usage:
        bundle = SolverBundle(gcn, node_mlp, device)
        solve_instance(path, bundle.branch_rule, bundle.node_selector)
        bundle.reset()  # between instances
    """

    def __init__(self, gcn, node_mlp=None, device='cpu', mode=None):
        self.cache      = EmbeddingCache()
        self.br         = create_learned_branchrule(gcn, self.cache, device)
        self.ns         = (create_neural_uct_selector(node_mlp, self.cache)
                           if node_mlp is not None
                           else create_default_selector(self.cache))

    def reset(self):
        """Call between instances to clear visit counts and cache."""
        self.cache.clear()
        if hasattr(self.ns, 'reset'):
            self.ns.reset()
        if hasattr(self.br, 'reset_stats'):
            self.br.reset_stats()


# ── Node data collection pass ──────────────────────────────────────────────────

class NodeDataCollector:
    """
    Runs instances with trained GCN brancher and records node selection data.
    Used to generate training data for NodeSelectorMLP.
    """

    def __init__(self, gcn, device='cpu'):
        self.gcn    = gcn
        self.device = device

    def collect_from_instance(self, instance_path, time_limit=300):
        """
        Solve one instance tracking node selection events.
        Returns trajectory dict with 'node_features' and 'optimal_path'.
        """
        from pyscipopt import Model, Branchrule, Nodesel, SCIP_RESULT

        cache = EmbeddingCache()
        node_features_log = []  # (features, node_number) per node selection event
        optimal_path_nodes = set()

        class LoggingNodeSel(Nodesel):
            def __init__(self_inner, gcn, cache):
                super().__init__()
                self_inner.gcn   = gcn
                self_inner.cache = cache

            def nodeselect(self_inner):
                # Log features for each open node
                children = list(self_inner.model.getChildren())
                siblings = list(self_inner.model.getSiblings())
                leaves   = list(self_inner.model.getLeaves())
                all_nodes = children + siblings + leaves

                root_lb = self_inner.model.getLowerbound()
                cutoff  = self_inner.model.getCutoffbound()
                max_d   = max((n.getDepth() for n in all_nodes), default=1)

                for node in all_nodes:
                    nn  = node.getNumber()
                    emb = self_inner.cache.get(nn)
                    feats = build_node_features(
                        node_embedding=emb,
                        lowerbound=node.getLowerbound(),
                        root_lowerbound=root_lb,
                        cutoff=cutoff if cutoff < 1e19 else root_lb + 1.0,
                        depth=node.getDepth(),
                        max_depth=max_d,
                        frac_sum=0.0,
                        n_cols=1,
                        node_visits=1,
                        parent_visits=1,
                    )
                    node_features_log.append((feats, nn))

                # Default: return best child
                selnode = self_inner.model.getPrioChild()
                if selnode is None:
                    selnode = self_inner.model.getBestLeaf()
                return {"selnode": selnode}

            def nodecomp(self_inner, n1, n2):
                return 0

        from branching.branch_rule import LearnedBranchRule
        model = Model()
        model.hideOutput(True)
        model.setParam("limits/time", time_limit)
        model.readProblem(instance_path)

        br = LearnedBranchRule(self.gcn, cache, self.device)
        ns = LoggingNodeSel(self.gcn, cache)

        model.includeBranchrule(br, "gcn_br", "", priority=1_000_000, maxdepth=-1, maxbounddist=1.0)
        model.includeNodesel(ns, "log_ns", "", stdpriority=1_000_000, memsavepriority=0)
        model.optimize()

        # Mark nodes on path to best solution
        # Simplified: mark all nodes that were explored as "on path" if solved optimally
        if model.getStatus() == 'optimal':
            # In a full implementation, track which nodes led to the optimal solution
            # Here we use a proxy: nodes at depth ≤ avg depth
            if node_features_log:
                avg_depth = np.mean([
                    f[0][cfg.EMBEDDING_DIM + 1] * 100   # depth_norm * 100 approx
                    for f in node_features_log
                ])
                optimal_path_nodes = set()  # TODO: proper tracking

        return {
            'node_features': node_features_log,
            'optimal_path':  optimal_path_nodes,
        }


# ── Main evaluation ────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    pt     = args.problem
    diff   = args.difficulty
    device = args.device

    print(f"\n{'='*60}")
    print(f"  Evaluating on {pt} ({diff}) — {args.n_instances} instances × {args.n_seeds} seeds")
    print(f"{'='*60}")

    paths  = get_instance_paths(pt, diff, args.n_instances)
    if not paths:
        return

    gcn     = load_gcn(pt, device)
    node_mlp = load_node_mlp(pt, device)

    # ── Collect node selection training data ───────────────────────────────────
    if args.collect_node_data:
        print("\nCollecting node selection training data...")
        collector    = NodeDataCollector(gcn, device)
        trajectories = []

        for i, path in enumerate(paths[:20]):
            print(f"  Instance {i+1}/20: {os.path.basename(path)}")
            traj = collector.collect_from_instance(path)
            trajectories.append(traj)

        feats, labels = build_node_training_labels(trajectories)
        out_path = os.path.join(cfg.DATA_DIR, f"{pt}_node_data.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump((feats, labels), f)
        print(f"Saved {len(feats)} node samples → {out_path}")
        print(f"  Positive rate: {labels.mean():.2%}")
        return

    # ── Policy 1: SCIP default ─────────────────────────────────────────────────
    print("\n[1/3] SCIP default (relpcost + hybridestim)...")
    default_agg = evaluate_policy(
        paths,
        branchrule_factory=lambda: None,
        nodesel_factory=lambda: None,
        n_seeds=args.n_seeds,
        time_limit=cfg.EVAL_TIME_LIMIT
    )
    print_results("SCIP Default", default_agg)

    # ── Policy 2: GCN branching + SCIP node selection ─────────────────────────
    print("\n[2/3] GCN branching + SCIP node selection...")

    def gcn_only_factory():
        # Each call creates a fresh bundle; no node MLP = default node sel
        b = SolverBundle(gcn, node_mlp=None, device=device)
        return b.br, b.ns

    gcn_only_agg = evaluate_policy(
        paths,
        branchrule_factory=lambda: SolverBundle(gcn, None, device).br,
        nodesel_factory=None,
        n_seeds=args.n_seeds,
        time_limit=cfg.EVAL_TIME_LIMIT
    )
    print_results("GCN + SCIP Node Sel", gcn_only_agg)

    # ── Policy 3: GCN branching + Neural UCT node selection ───────────────────
    if node_mlp is not None:
        print("\n[3/3] GCN branching + Neural UCT node selection...")

        # IMPORTANT: br and ns MUST share the same cache instance.
        # We create a list of pre-built bundles (one per solve).
        n_solves = len(paths) * args.n_seeds
        bundles  = [SolverBundle(gcn, node_mlp, device) for _ in range(n_solves)]
        bundle_iter = iter(bundles)

        full_results = []
        for path in paths:
            for seed in range(args.n_seeds):
                bundle = next(bundle_iter)
                bundle.reset()
                from utils.metrics import solve_instance
                r = solve_instance(path, bundle.br, bundle.ns,
                                   cfg.EVAL_TIME_LIMIT, seed)
                r['instance'] = path
                r['seed']     = seed
                full_results.append(r)

        from utils.metrics import aggregate_results
        full_agg = aggregate_results(full_results, cfg.EVAL_TIME_LIMIT)
        print_results("GCN + Neural UCT", full_agg)

        # ── GO/NO GO comparison ────────────────────────────────────────────────
        print("\n\nGO/NO GO: GCN+NeuralUCT vs GCN+DefaultNodeSel")
        compare_policies(gcn_only_agg, full_agg)

        print("\nGO/NO GO: GCN+NeuralUCT vs SCIP Default")
        compare_policies(default_agg, full_agg)
    else:
        print("\n[3/3] Skipped — no NodeMLP checkpoint found")
        print("  Run: python evaluate.py --collect-node-data")
        print("  Then: python train.py --skip-generate --skip-collect --skip-gcn")

    print("\nDone.")


if __name__ == "__main__":
    main()
