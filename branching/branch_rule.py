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
        Uses npriocands (priority integer/binary candidates) not ncands.
        Falls back to SCIP default if learned policy fails.
        """
        result = self.model.getLPBranchCands()
        if result is None:
            return None

        cands, cand_sols, cand_fracs, ncands, npriocands, nimplcands = result

        # Use priority candidates only (paper and PySCIPOpt docs recommend this)
        n_prio = npriocands if npriocands > 0 else ncands
        if n_prio == 0:
            return None

        prio_cands = cands[:n_prio]
        prio_fracs = cand_fracs[:n_prio]

        if not self._use_learned:
            return None

        graph = extract_bipartite_graph(self.model)
        if graph is None or not graph["cand_mask"].any():
            return None

        # Store frac_sum using priority candidates only
        graph["frac_sum"] = float(sum(min(f, 1.0-f) for f in prio_fracs))
        graph["n_cols"]   = len(self.model.getLPColsData())

        logits, var_embeddings = self._run_gcn(graph)
        if logits is None:
            return None

        # cand_mask marks all LP branch candidates in column order.
        # logits[i] corresponds to the i-th True position in cand_mask.
        # Map argmax logit → column index → variable name → find in prio_cands.
        cols = self.model.getLPColsData()
        cand_col_indices = np.where(graph["cand_mask"])[0]  # shape (n_cands_in_mask,)

        if len(cand_col_indices) == 0:
            return None

        # Restrict logits to priority candidates only:
        # Build set of priority candidate names for fast lookup
        prio_names = {v.name for v in prio_cands}
        prio_mask_local = []   # indices into cand_col_indices that are prio candidates
        for local_i, col_i in enumerate(cand_col_indices):
            if col_i < len(cols) and cols[col_i].getVar().name in prio_names:
                prio_mask_local.append(local_i)

        if not prio_mask_local:
            # Fallback: use all candidates
            prio_mask_local = list(range(len(cand_col_indices)))

        # Pick best among priority candidates
        prio_logits = logits[prio_mask_local]
        best_prio_local = prio_mask_local[int(prio_logits.argmax())]
        best_col_idx    = cand_col_indices[best_prio_local]

        # Match by name (robust to ordering differences between getLPColsData and getLPBranchCands)
        if best_col_idx < len(cols):
            best_name = cols[best_col_idx].getVar().name
            chosen_var = None
            for v in prio_cands:
                if v.name == best_name:
                    chosen_var = v
                    break
            if chosen_var is None:
                chosen_var = prio_cands[0]   # safe fallback
        else:
            chosen_var = prio_cands[0]

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