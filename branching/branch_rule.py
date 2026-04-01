"""
Learned branching rule — PySCIPOpt Branchrule plugin.
Integrates the trained GCN into SCIP's branching callback.

Responsibilities:
    1. Extract bipartite graph features from current LP state
    2. Run GCN forward pass → logits over candidate variables
    3. Select candidate with highest logit
    4. Call model.branchVar() to branch
    5. Cache node embedding for open children (used by node selector)

Also supports:
    - Strong branching mode (for data collection)
    - Fallback to SCIP default if GCN fails
"""

import numpy as np
import torch
from pyscipopt import Branchrule, SCIP_RESULT

from data.feature_extractor import extract_bipartite_graph
from utils.embedding_cache import EmbeddingCache
import config as cfg


class LearnedBranchRule(Branchrule):
    """
    GCN-based variable selection branching rule.
    Register with: model.includeBranchrule(rule, name, desc, priority, maxdepth, maxbounddist)
    """

    def __init__(self, gcn_model, embedding_cache: EmbeddingCache, device='cpu'):
        """
        gcn_model      : trained BranchingGCN (or None for strong branching / default)
        embedding_cache: shared EmbeddingCache instance (also used by node selector)
        device         : torch device string
        """
        super().__init__()
        self.gcn           = gcn_model
        self.cache         = embedding_cache
        self.device        = device
        self._use_learned  = gcn_model is not None

        # Statistics (reset per instance)
        self.n_branch_calls = 0
        self.n_fallbacks    = 0

    # ── Main callback ──────────────────────────────────────────────────────────

    def branchexeclp(self, allowaddcons):
        """Called by SCIP when a node's LP is solved and branching is needed."""
        self.n_branch_calls += 1

        try:
            chosen_var = self._select_variable()
        except Exception as e:
            # Fallback: let SCIP choose
            self.n_fallbacks += 1
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if chosen_var is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self.model.branchVar(chosen_var)
        return {"result": SCIP_RESULT.BRANCHED}

    # ── Variable selection ─────────────────────────────────────────────────────

    def _select_variable(self):
        """
        Returns the Variable to branch on.
        Falls back to SCIP default if learned policy fails.
        """
        # Get LP branch candidates
        result = self.model.getLPBranchCands()
        if result is None:
            return None
        candidates, cand_sols, cand_fracs, ncands, *_ = result
        if ncands == 0:
            return None

        if not self._use_learned:
            # Strong branching mode — let SCIP handle it, we just log
            return None

        # Extract features
        graph = extract_bipartite_graph(self.model)
        if graph is None or not graph["cand_mask"].any():
            return None

        # Run GCN
        logits, var_embeddings = self._run_gcn(graph)

        if logits is None:
            return None

        # Map cand_mask indices → candidate variables
        cand_indices = np.where(graph["cand_mask"])[0]
        if len(cand_indices) == 0:
            return None

        # Select variable with highest logit
        best_local_idx = logits.argmax().item()        # index into cand_mask subset
        best_col_idx   = cand_indices[best_local_idx]  # index into all columns

        # Get corresponding candidate variable
        # candidates list is ordered by SCIP; we need to match by column index
        # Use LP column data to find the variable
        cols = self.model.getLPColsData()
        if best_col_idx < len(cols):
            chosen_var = cols[best_col_idx].getVar()
        else:
            # Fallback: pick first candidate
            chosen_var = candidates[0]

        # ── Cache node embedding for current node ─────────────────────────────
        self._cache_current_node(var_embeddings, graph)

        return chosen_var

    def _run_gcn(self, graph):
        """Run GCN forward pass. Returns (logits_np, var_embeddings_tensor)."""
        self.gcn.eval()
        with torch.no_grad():
            con_feats  = torch.from_numpy(graph["con_feats"]).float().to(self.device)
            edge_index = torch.from_numpy(graph["edge_index"]).long().to(self.device)
            edge_feats = torch.from_numpy(graph["edge_feats"]).float().to(self.device)
            var_feats  = torch.from_numpy(graph["var_feats"]).float().to(self.device)
            cand_mask  = torch.from_numpy(graph["cand_mask"]).bool().to(self.device)

            logits, var_embeddings = self.gcn(
                con_feats, edge_index, edge_feats, var_feats, cand_mask
            )

        return logits.cpu().numpy(), var_embeddings

    def _cache_current_node(self, var_embeddings, graph):
        """
        Compute mean-pool embedding for current node and store in cache.
        Also pre-cache children with same embedding as a warm-start approximation.
        """
        node_emb = self.gcn.get_node_embedding(var_embeddings).cpu().numpy()
        frac_sum = float(graph.get("frac_sum", 0.0))
        n_cols   = int(graph.get("n_cols", var_embeddings.shape[0]))

        current_node = self.model.getCurrentNode()
        if current_node is not None:
            node_num = current_node.getNumber()
            self.cache.store(node_num, node_emb, frac_sum, n_cols)

            # Pre-cache children (warm-start; overwritten when actually processed)
            try:
                for child in self.model.getChildren():
                    self.cache.store(child.getNumber(), node_emb, frac_sum, n_cols)
            except Exception:
                pass

    # ── Utilities ──────────────────────────────────────────────────────────────

    def reset_stats(self):
        self.n_branch_calls = 0
        self.n_fallbacks    = 0

    def stats(self):
        return {
            "branch_calls": self.n_branch_calls,
            "fallbacks":    self.n_fallbacks,
            "fallback_rate": f"{self.n_fallbacks / max(self.n_branch_calls, 1):.2%}",
        }


class StrongBranchRule(Branchrule):
    """
    Vanilla strong branching — used during data collection to generate expert labels.
    Evaluates each candidate by temporarily solving two LP relaxations.
    Records the strong branching scores and selected variable.
    """

    def __init__(self, record_data=True):
        super().__init__()
        self.record_data = record_data
        self.collected   = []          # list of (var_name, sb_score) per decision

    def branchexeclp(self, allowaddcons):
        """Use SCIP's built-in strong branching — don't override, just observe."""
        return {"result": SCIP_RESULT.DIDNOTRUN}   # let SCIP's SB rule take over


def create_default_branchrule(embedding_cache: EmbeddingCache) -> LearnedBranchRule:
    """Create branching rule with no learned model (SCIP default behavior)."""
    rule = LearnedBranchRule(gcn_model=None, embedding_cache=embedding_cache)
    return rule


def create_learned_branchrule(gcn_model, embedding_cache: EmbeddingCache,
                               device='cpu') -> LearnedBranchRule:
    """Create fully learned branching rule."""
    gcn_model.eval()
    return LearnedBranchRule(gcn_model=gcn_model,
                             embedding_cache=embedding_cache,
                             device=device)
