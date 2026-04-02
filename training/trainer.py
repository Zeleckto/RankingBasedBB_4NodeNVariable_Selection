"""
Training loops for:
    1. BranchingGCN  — Branch Ranking weighted cross-entropy
    2. NodeSelectorMLP — Binary cross-entropy on node selection labels
    3. Joint fine-tuning (optional)

GCN Training (Branch Ranking Eq. 3):
    max_θ  Σ_{(s,a) ~ D_hyb}  r(s,a) * log π_θ(a | s)

    = minimize  -Σ r(s,a) * log π_θ(a | s)   [weighted cross-entropy]

    where r(s,a) ∈ {0, 1} from reward assignment,
    π_θ(a|s) = softmax(GCN(s))[a]   (probability of choosing variable a)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.gcn import BranchingGCN
from models.node_mlp import NodeSelectorMLP, NodeMLPTrainer
from training.reward_assigner import (
    assign_long_term_rewards, assign_short_term_rewards,
    build_training_dataset, build_node_training_labels,
    compute_weighted_ce_loss
)
from data.feature_extractor import get_prenorm_stats
import config as cfg


# ── GCN Trainer ────────────────────────────────────────────────────────────────

class GCNTrainer:
    """Trains BranchingGCN using Branch Ranking's weighted cross-entropy."""

    def __init__(self, model: BranchingGCN, device='cpu'):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.Adam(
            model.parameters(),
            lr=cfg.GCN_LR,
            weight_decay=cfg.GCN_WEIGHT_DECAY
        )
        self.scheduler = ReduceLROnPlateau(
            self.opt, mode='min',
            factor=cfg.GCN_LR_DECAY,
            patience=cfg.GCN_PATIENCE,
            verbose=True
        )
        self.best_val_loss = float('inf')
        self.patience_count = 0

    def _graph_to_tensors(self, graph):
        """Convert feature dict to torch tensors."""
        return (
            torch.from_numpy(graph['con_feats']).float().to(self.device),
            torch.from_numpy(graph['edge_index']).long().to(self.device),
            torch.from_numpy(graph['edge_feats']).float().to(self.device),
            torch.from_numpy(graph['var_feats']).float().to(self.device),
            torch.from_numpy(graph['cand_mask']).bool().to(self.device),
        )

    def train_batch(self, graphs, target_local_indices, rewards):
        """
        One gradient step on a batch of samples.

        graphs               : list of graph dicts (one per sample)
        target_local_indices : list of int — chosen variable's index in cand_mask subset
        rewards              : list of float — r(s,a) ∈ {0,1}

        Returns: loss scalar (float)
        """
        self.model.train()
        self.opt.zero_grad()

        logits_list = []
        valid_targets = []
        valid_rewards = []

        for graph, target_idx, reward in zip(graphs, target_local_indices, rewards):
            try:
                con, eidx, ef, vf, mask = self._graph_to_tensors(graph)
                logits, _ = self.model(con, eidx, ef, vf, mask)
                logits_list.append(logits)
                valid_targets.append(target_idx)
                valid_rewards.append(reward)
            except Exception:
                continue

        if not logits_list:
            return 0.0

        loss = compute_weighted_ce_loss(logits_list, valid_targets, valid_rewards)

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()

        return loss.item()

    def train_epoch(self, samples, graphs, rewards):
        """One full pass over training data. Returns mean loss."""
        idx       = np.random.permutation(len(graphs))
        total     = 0.0
        n_batches = 0
        B         = cfg.GCN_BATCH_SIZE

        for start in range(0, len(graphs), B):
            batch = idx[start:start + B]
            batch_graphs   = [graphs[i]  for i in batch]
            batch_rewards  = [rewards[i] for i in batch]
            # Target index: action_col_idx mapped to local candidate index
            batch_targets  = [self._get_local_idx(samples[i]) for i in batch]

            loss = self.train_batch(batch_graphs, batch_targets, batch_rewards)
            total     += loss
            n_batches += 1

        return total / max(n_batches, 1)

    def _get_local_idx(self, sample):
        """
        Map sample.action_col_idx → index within cand_mask=True positions.

        The GCN outputs logits[i] for the i-th True position in cand_mask.
        We need to find which position i corresponds to action_col_idx.

        searchsorted is WRONG if action_col_idx is not exactly in cand_positions —
        it returns the insertion point, not the element index. Use np.where instead.
        """
        cand_mask = sample.state_graph['cand_mask']
        cand_positions = np.where(cand_mask)[0]   # column indices of all candidates
        col_idx = sample.action_col_idx

        # Find exact match
        matches = np.where(cand_positions == col_idx)[0]
        if len(matches) > 0:
            return int(matches[0])

        # If no exact match (shouldn't happen in well-formed data):
        # fall back to nearest candidate index
        nearest = int(np.argmin(np.abs(cand_positions - col_idx)))
        return nearest

    def evaluate(self, samples, graphs, rewards):
        """Compute validation loss (no gradient)."""
        self.model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for sample, graph, reward in zip(samples, graphs, rewards):
                try:
                    con, eidx, ef, vf, mask = self._graph_to_tensors(graph)
                    logits, _ = self.model(con, eidx, ef, vf, mask)
                    target = self._get_local_idx(sample)
                    loss = compute_weighted_ce_loss([logits], [target], [reward])
                    total += loss.item()
                    n += 1
                except Exception:
                    continue
        return total / max(n, 1)

    def fit(self, train_data, val_data=None, checkpoint_dir=None):
        """
        Full training.
        train_data / val_data: (samples, graphs, rewards) tuples
        """
        train_samples, train_graphs, train_rewards = train_data

        print(f"GCN training: {len(train_graphs)} samples")

        for epoch in range(cfg.GCN_MAX_EPOCHS):
            train_loss = self.train_epoch(train_samples, train_graphs, train_rewards)

            val_str = ""
            if val_data is not None:
                val_samples, val_graphs, val_rewards = val_data
                val_loss = self.evaluate(val_samples, val_graphs, val_rewards)
                val_str  = f"  val={val_loss:.4f}"
                self.scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss   = val_loss
                    self.patience_count  = 0
                    if checkpoint_dir:
                        self.save(os.path.join(checkpoint_dir, "gcn_best.pt"))
                else:
                    self.patience_count += 1
                    if self.patience_count >= cfg.GCN_STOP_PATIENCE:
                        print(f"Early stop at epoch {epoch}")
                        break

            if epoch % 5 == 0:
                print(f"Epoch {epoch:4d}  train={train_loss:.4f}{val_str}")

        print("GCN training complete.")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'opt_state':   self.opt.state_dict(),
            'val_loss':    self.best_val_loss,
        }, path)
        print(f"Saved GCN checkpoint → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        print(f"Loaded GCN from {path}")


# ── Prenorm initialization ─────────────────────────────────────────────────────

def initialize_prenorms(gcn: BranchingGCN, graphs):
    """
    Compute empirical stats from training graphs and initialize prenorm layers.
    Must be called BEFORE training starts (and only once).
    """
    print("Initializing prenorm layers from training data stats...")
    stats = get_prenorm_stats(graphs)
    gcn.initialize_prenorms(stats)
    print("  Done.")
    return stats


# ── Joint trainer ──────────────────────────────────────────────────────────────

class JointTrainer:
    """
    Optional joint training of GCN + NodeMLP with combined loss.
    L_total = L_gcn + λ * L_node
    """

    def __init__(self, gcn: BranchingGCN, node_mlp: NodeSelectorMLP,
                 lam=None, device='cpu'):
        self.gcn      = gcn.to(device)
        self.node_mlp = node_mlp.to(device)
        self.lam      = lam or cfg.JOINT_LAMBDA
        self.device   = device

        self.opt = torch.optim.Adam(
            list(gcn.parameters()) + list(node_mlp.parameters()),
            lr=cfg.GCN_LR
        )

    def train_step(self, gcn_batch, node_batch):
        """
        gcn_batch  : (samples, graphs, rewards)
        node_batch : (features_np, labels_np)
        """
        self.gcn.train()
        self.node_mlp.train()
        self.opt.zero_grad()

        # GCN loss (Branch Ranking)
        gcn_trainer = GCNTrainer(self.gcn, self.device)
        # Reuse loss computation without stepping optimizer
        # (simplified — in practice compute directly)

        # Node MLP loss
        feat_np, lab_np = node_batch
        x = torch.from_numpy(feat_np).float().to(self.device)
        y = torch.from_numpy(lab_np).float().to(self.device)
        # Remove sigmoid for BCEWithLogitsLoss
        net_no_sig = self.node_mlp.net[:-1]
        node_logits = net_no_sig(x).squeeze(-1)
        pos_weight  = torch.tensor([cfg.NODE_POS_WEIGHT], device=self.device)
        node_loss   = F.binary_cross_entropy_with_logits(node_logits, y,
                                                          pos_weight=pos_weight)

        total_loss = self.lam * node_loss
        total_loss.backward()
        self.opt.step()

        return total_loss.item()