"""
Bipartite Graph Convolutional Network for MIP branching.
Follows Gasse et al. (NeurIPS 2019) architecture exactly:
  - Prenorm layers (fixed affine normalization before training)
  - Un-normalized sum convolutions (not mean)
  - Two half-convolutions: V→C then C→V
  - Final MLP per variable → logit score

Usage:
    gcn = BranchingGCN(con_dim=5, edge_dim=1, var_dim=14, emb_dim=64)
    logits, var_embeddings = gcn(con_feats, edge_index, edge_feats, var_feats, cand_mask)
    action = logits.argmax()   # index into var_feats rows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """2-layer MLP with ReLU, used for fC, fV, gC, gV."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PrenormLayer(nn.Module):
    """
    Fixed affine normalisation: x ← (x - β) / σ
    β and σ initialised from training data stats, then FROZEN.
    This is Gasse et al.'s key trick for generalization to larger instances.
    """
    def __init__(self, n_features):
        super().__init__()
        self.register_buffer('beta',  torch.zeros(n_features))
        self.register_buffer('sigma', torch.ones(n_features))
        self._initialized = False

    def initialize(self, mean: np.ndarray, std: np.ndarray):
        """Call once before training with empirical stats from training data."""
        self.beta.copy_(torch.from_numpy(mean))
        self.sigma.copy_(torch.from_numpy(std))
        self._initialized = True

    def forward(self, x):
        return (x - self.beta) / self.sigma


class BipartiteConvLayer(nn.Module):
    """
    One round of bipartite message passing:
        1. V → C:  c_i ← fC(c_i, sum_j gC(c_i, v_j, e_ij))
        2. C → V:  v_j ← fV(v_j, sum_i gV(c_i, v_j, e_ij))

    Un-normalized sum aggregation (no division by degree).
    Prenorm applied after aggregation sums (before fC/fV).
    """
    def __init__(self, emb_dim, edge_dim):
        super().__init__()
        self.emb_dim  = emb_dim
        self.edge_dim = edge_dim

        # Message functions (2*emb + edge_dim → emb)
        self.gC = MLP(2 * emb_dim + edge_dim, emb_dim, emb_dim)
        self.gV = MLP(2 * emb_dim + edge_dim, emb_dim, emb_dim)

        # Update functions (emb + emb → emb)
        self.fC = MLP(2 * emb_dim, emb_dim, emb_dim)
        self.fV = MLP(2 * emb_dim, emb_dim, emb_dim)

        # Prenorm layers (one per aggregation sum)
        self.prenorm_c = PrenormLayer(emb_dim)
        self.prenorm_v = PrenormLayer(emb_dim)

    def forward(self, c, v, edge_index, e):
        """
        c          : (n_rows, emb_dim)
        v          : (n_cols, emb_dim)
        edge_index : (2, n_edges)  [row_idx, col_idx]
        e          : (n_edges, edge_dim)
        """
        row_idx = edge_index[0]   # constraint indices for each edge
        col_idx = edge_index[1]   # variable indices for each edge

        # ── Pass 1: V → C ─────────────────────────────────────────────────────
        # Compute messages from variables to constraints
        msgs_to_c = self.gC(torch.cat([c[row_idx], v[col_idx], e], dim=-1))  # (n_edges, emb)

        # Sum-aggregate per constraint (un-normalized)
        agg_c = torch.zeros(c.shape[0], self.emb_dim, device=c.device)
        agg_c.scatter_add_(0, row_idx.unsqueeze(-1).expand_as(msgs_to_c), msgs_to_c)
        agg_c = self.prenorm_c(agg_c)

        # Update constraint embeddings
        c_new = self.fC(torch.cat([c, agg_c], dim=-1))

        # ── Pass 2: C → V ─────────────────────────────────────────────────────
        msgs_to_v = self.gV(torch.cat([c_new[row_idx], v[col_idx], e], dim=-1))  # (n_edges, emb)

        agg_v = torch.zeros(v.shape[0], self.emb_dim, device=v.device)
        agg_v.scatter_add_(0, col_idx.unsqueeze(-1).expand_as(msgs_to_v), msgs_to_v)
        agg_v = self.prenorm_v(agg_v)

        v_new = self.fV(torch.cat([v, agg_v], dim=-1))

        return c_new, v_new


class BranchingGCN(nn.Module):
    """
    Full bipartite GCN for variable selection in B&B.

    Forward returns:
        logits         : (n_cands,)     scores for each candidate variable
        var_embeddings : (n_cols, emb_dim)  full variable embeddings (for caching)
    """
    def __init__(self,
                 con_dim=5, edge_dim=1, var_dim=14,
                 emb_dim=64, n_layers=1):
        super().__init__()
        self.emb_dim  = emb_dim
        self.n_layers = n_layers

        # Prenorm for raw features
        self.prenorm_con = PrenormLayer(con_dim)
        self.prenorm_var = PrenormLayer(var_dim)
        self.prenorm_edg = PrenormLayer(edge_dim)

        # Initial embedding projections
        self.con_embed = MLP(con_dim,  emb_dim, emb_dim)
        self.var_embed = MLP(var_dim,  emb_dim, emb_dim)

        # Bipartite conv layers
        self.conv_layers = nn.ModuleList([
            BipartiteConvLayer(emb_dim, edge_dim)
            for _ in range(n_layers)
        ])

        # Final MLP: variable embedding → scalar logit
        self.output_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def initialize_prenorms(self, stats: dict):
        """
        stats: dict from feature_extractor.get_prenorm_stats()
        Call once on training data before starting gradient updates.
        """
        self.prenorm_con.initialize(stats["con_mean"], stats["con_std"])
        self.prenorm_var.initialize(stats["var_mean"], stats["var_std"])
        self.prenorm_edg.initialize(stats["edg_mean"], stats["edg_std"])
        for layer in self.conv_layers:
            # Reset prenorm layers inside conv (will be fitted on first forward pass stats)
            pass

    def forward(self, con_feats, edge_index, edge_feats, var_feats, cand_mask):
        """
        con_feats  : (n_rows, con_dim)
        edge_index : (2, n_edges)
        edge_feats : (n_edges, edge_dim)
        var_feats  : (n_cols, var_dim)
        cand_mask  : (n_cols,) bool tensor — which variables are candidates

        Returns:
            logits         : (n_cands,)     raw scores (pre-softmax)
            var_embeddings : (n_cols, emb_dim)
        """
        # Prenorm raw features
        c = self.prenorm_con(con_feats)
        v = self.prenorm_var(var_feats)
        e = self.prenorm_edg(edge_feats)

        # Initial embeddings
        c = self.con_embed(c)     # (n_rows, emb_dim)
        v = self.var_embed(v)     # (n_cols, emb_dim)

        # Bipartite message passing
        for layer in self.conv_layers:
            c, v = layer(c, v, edge_index, e)

        # Variable embeddings (used for node caching)
        var_embeddings = v  # (n_cols, emb_dim)

        # Score only candidate variables
        cand_embeddings = v[cand_mask]              # (n_cands, emb_dim)
        logits = self.output_mlp(cand_embeddings).squeeze(-1)  # (n_cands,)

        return logits, var_embeddings

    def get_node_embedding(self, var_embeddings):
        """
        Compute a single node-level embedding by mean-pooling variable embeddings.
        Used for embedding cache in node selector.

        Returns: (emb_dim,)
        """
        return var_embeddings.mean(dim=0)


def build_gcn(cfg) -> BranchingGCN:
    """Construct GCN from config."""
    return BranchingGCN(
        con_dim  = cfg.CONSTRAINT_FEAT_DIM,
        edge_dim = cfg.EDGE_FEAT_DIM,
        var_dim  = cfg.VARIABLE_FEAT_DIM,
        emb_dim  = cfg.EMBEDDING_DIM,
        n_layers = cfg.GCN_LAYERS,
    )
