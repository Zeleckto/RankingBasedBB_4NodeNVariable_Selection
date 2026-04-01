"""
Node Selector — PySCIPOpt Nodesel plugin.

Modes (set via NODE_SEL_MODE in config):
    "default"    : Always return SCIP's best node (hybridestim behavior via getPrioChild/getBestNode).
                   Use this until MLP is trained.
    "neural_uct" : UCT score with LEARNED MLP scoring replacing the fixed formula.
                   score(node) = MLP([cached_embedding, lb_norm, depth, frac, visit_ratio])
                   Adds visit count tracking on top of MLP score.

The mode is checked at construction time — swap without changing anything else.

Register with:
    sel = NeuralUCTNodeSelector(mlp_model, embedding_cache, mode="neural_uct")
    scip.includeNodesel(sel, "neural_uct", "Neural UCT Node Selector",
                        stdpriority=1_000_000, memsavepriority=0)
"""

import numpy as np
from pyscipopt import Nodesel

from utils.embedding_cache import EmbeddingCache
from models.node_mlp import NodeSelectorMLP, build_node_features
import config as cfg


class NeuralUCTNodeSelector(Nodesel):
    """
    Unified node selector supporting default and neural UCT modes.

    Key data structures:
        visit_counts : dict node_number → int   (how many times this subtree visited)
        max_depth    : int                       (running max depth seen so far)
    """

    def __init__(self,
                 mlp_model: NodeSelectorMLP = None,
                 embedding_cache: EmbeddingCache = None,
                 mode: str = None):
        super().__init__()
        self.mlp    = mlp_model
        self.cache  = embedding_cache
        self.mode   = mode or cfg.NODE_SEL_MODE

        # UCT visit tracking
        self.visit_counts = {}    # node_number → int
        self.max_depth    = 1
        self._n_selections = 0
        self._prune_every  = 200  # prune dead cache entries every N selections

    # ── Required callbacks ──────────────────────────────────────────────────────

    def nodeselect(self):
        """
        Main callback: return the next node to process.
        Called once per B&B iteration.
        """
        self._n_selections += 1

        if self.mode == "default":
            return self._default_select()

        if self.mode == "neural_uct":
            return self._neural_uct_select()

        # Fallback
        return self._default_select()

    def nodecomp(self, node1, node2):
        """
        Comparison function for SCIP's internal priority queue.
        Returns: <0 if node1 better, 0 if equal, >0 if node2 better.
        We delegate to lowerbound for compatibility.
        """
        lb1 = node1.getLowerbound()
        lb2 = node2.getLowerbound()
        if lb1 < lb2:
            return -1
        elif lb1 > lb2:
            return 1
        return 0

    # ── Selection strategies ───────────────────────────────────────────────────

    def _default_select(self):
        """
        Mirrors SCIP hybridestim: try prio child → prio sibling → best leaf.
        Zero learning, just a safe default.
        """
        selnode = self.model.getPrioChild()
        if selnode is None:
            selnode = self.model.getPrioSibling()
        if selnode is None:
            selnode = self.model.getBestLeaf()
        return {"selnode": selnode}

    def _neural_uct_select(self):
        """
        Score all open nodes using MLP + UCT visit term, return argmax.
        Falls back to default if MLP is None or scoring fails.
        """
        if self.mlp is None:
            return self._default_select()

        try:
            all_nodes = self._get_all_open_nodes()
            if not all_nodes:
                return self._default_select()

            scores = self._score_nodes(all_nodes)
            best_idx  = int(np.argmax(scores))
            best_node = all_nodes[best_idx]

            # Update visit counts along path to best node
            self._update_visits(best_node)

            # Periodic cache pruning
            if self._n_selections % self._prune_every == 0 and self.cache is not None:
                active = {n.getNumber() for n in all_nodes}
                self.cache.prune(active)

            return {"selnode": best_node}

        except Exception:
            return self._default_select()

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_nodes(self, nodes) -> np.ndarray:
        """
        Compute MLP score for each node.
        Returns (n_nodes,) float array — higher is better.
        """
        try:
            root_lb = self.model.getLowerboundRoot()
        except AttributeError:
            root_lb = self.model.getLowerbound()
        cutoff    = self.model.getCutoffbound()
        max_depth = max(self.max_depth, 1)

        feature_rows = []
        for node in nodes:
            node_num = node.getNumber()
            lb       = node.getLowerbound()
            depth    = node.getDepth()
            self.max_depth = max(self.max_depth, depth)

            # Embedding + frac_sum + n_cols from shared cache
            if self.cache is not None:
                emb, frac_sum, n_cols = self.cache.get(node_num)
            else:
                emb      = np.zeros(cfg.EMBEDDING_DIM, dtype=np.float32)
                frac_sum = 0.0
                n_cols   = 1

            # UCT visit counts
            node_visits   = self.visit_counts.get(node_num, 1)
            parent        = node.getParent()
            parent_visits = self.visit_counts.get(parent.getNumber(), 1) \
                            if parent is not None else 1

            feats = build_node_features(
                node_embedding  = emb,
                lowerbound      = lb,
                root_lowerbound = root_lb,
                cutoff          = cutoff if cutoff < 1e19 else lb + 1.0,
                depth           = depth,
                max_depth       = max_depth,
                frac_sum        = frac_sum,   # ← now correctly populated from cache
                n_cols          = max(n_cols, 1),
                node_visits     = node_visits,
                parent_visits   = parent_visits,
            )
            feature_rows.append(feats)

        feature_matrix = np.stack(feature_rows, axis=0)   # (n_nodes, 68)
        scores = self.mlp.score_nodes(feature_matrix)      # (n_nodes,)
        return scores

    # ── Visit tracking (UCT) ───────────────────────────────────────────────────

    def _update_visits(self, node):
        """
        Increment visit count for selected node and all its ancestors.
        This is the path update rule from UCT.
        """
        current = node
        while current is not None:
            num = current.getNumber()
            self.visit_counts[num] = self.visit_counts.get(num, 0) + 1
            try:
                current = current.getParent()
            except Exception:
                break

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_all_open_nodes(self):
        """Collect all open nodes: children + siblings + leaves."""
        nodes = []
        try:
            nodes += list(self.model.getChildren())
        except Exception:
            pass
        try:
            nodes += list(self.model.getSiblings())
        except Exception:
            pass
        try:
            nodes += list(self.model.getLeaves())
        except Exception:
            pass
        return nodes

    def reset(self):
        """Call between problem instances to clear state."""
        self.visit_counts.clear()
        self.max_depth   = 1
        self._n_selections = 0

    def stats(self):
        return {
            "mode":          self.mode,
            "n_selections":  self._n_selections,
            "nodes_tracked": len(self.visit_counts),
            "max_depth":     self.max_depth,
            "cache_stats":   self.cache.stats() if self.cache else "N/A",
        }


# ── Factory functions ──────────────────────────────────────────────────────────

def create_default_selector(embedding_cache: EmbeddingCache = None) -> NeuralUCTNodeSelector:
    """Default mode — no learning, mirrors hybridestim."""
    return NeuralUCTNodeSelector(mlp_model=None, embedding_cache=embedding_cache,
                                  mode="default")


def create_neural_uct_selector(mlp_model: NodeSelectorMLP,
                                embedding_cache: EmbeddingCache) -> NeuralUCTNodeSelector:
    """Neural UCT mode — MLP scoring + UCT visit term."""
    mlp_model.eval()
    return NeuralUCTNodeSelector(mlp_model=mlp_model, embedding_cache=embedding_cache,
                                  mode="neural_uct")
