"""
Node Selector MLP — scores open B&B nodes using cached GCN embeddings + scalar features.

Input features per node (68-dim):
    cached_embedding   (64)  : mean-pooled GCN variable embeddings from node creation
    lowerbound_norm    ( 1)  : (lb - root_lb) / (cutoff - root_lb)
    depth_norm         ( 1)  : depth / max_depth_seen
    frac_sum_norm      ( 1)  : sum of fractionalities / n_cols
    visit_ratio        ( 1)  : parent_visits / (node_visits + 1)  ← UCT exploration term

Output: scalar in [0,1] (probability that this node is on optimal path)

Training: Binary Cross-Entropy
    y=1 if node was ancestor of best primal solution in trajectory
    y=0 otherwise
    Positive class weight ~10x (optimal nodes are rare)
"""

import torch
import torch.nn as nn
import numpy as np
import config as cfg


class NodeSelectorMLP(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=None):
        super().__init__()
        input_dim   = input_dim   or cfg.NODE_INPUT_DIM
        hidden_dims = hidden_dims or cfg.NODE_HIDDEN_DIMS

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, input_dim) → (batch,) scores in [0,1]"""
        return self.net(x).squeeze(-1)

    def score_nodes(self, node_features: np.ndarray) -> np.ndarray:
        """
        Convenience: score a batch of nodes.
        node_features: (n_nodes, input_dim) numpy array
        Returns: (n_nodes,) numpy array of scores
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(node_features).float()
            return self.forward(x).numpy()


def build_node_features(node_embedding: np.ndarray,
                        lowerbound: float,
                        root_lowerbound: float,
                        cutoff: float,
                        depth: int,
                        max_depth: int,
                        frac_sum: float,
                        n_cols: int,
                        node_visits: int,
                        parent_visits: int) -> np.ndarray:
    """
    Build the 68-dim feature vector for one node.
    Called inside the node selector at each selection step.
    """
    emb = node_embedding  # (64,)

    # Normalize lowerbound: how close to cutoff?
    span = cutoff - root_lowerbound
    lb_norm = (lowerbound - root_lowerbound) / (span + 1e-8) if span > 1e-8 else 0.0
    lb_norm = float(np.clip(lb_norm, 0.0, 1.0))

    depth_norm = float(depth) / float(max(max_depth, 1))

    frac_norm = frac_sum / float(max(n_cols, 1))

    visit_ratio = float(parent_visits) / float(node_visits + 1)

    scalar_feats = np.array([lb_norm, depth_norm, frac_norm, visit_ratio], dtype=np.float32)

    return np.concatenate([emb, scalar_feats])   # (68,)


class NodeMLPTrainer:
    """
    Trains the NodeSelectorMLP from labeled node data.

    Training data format (per sample):
        features : np.ndarray (input_dim,)
        label    : int  1 = on optimal path, 0 = not
    """

    def __init__(self, model: NodeSelectorMLP, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.NODE_LR,
            weight_decay=cfg.NODE_WEIGHT_DECAY
        )
        pos_weight = torch.tensor([cfg.NODE_POS_WEIGHT], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Use BCEWithLogitsLoss — remove Sigmoid from model for training stability
        # (we add Sigmoid back via score_nodes for inference)

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        One epoch over the dataset.
        features : (N, input_dim)
        labels   : (N,) binary int
        Returns mean loss.
        """
        self.model.train()
        idx = np.random.permutation(len(features))
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(features), cfg.NODE_BATCH_SIZE):
            batch_idx  = idx[start:start + cfg.NODE_BATCH_SIZE]
            x = torch.from_numpy(features[batch_idx]).float().to(self.device)
            y = torch.from_numpy(labels[batch_idx]).float().to(self.device)

            self.optimizer.zero_grad()
            # Forward through net WITHOUT sigmoid (BCEWithLogitsLoss expects logits)
            logits = self._forward_logits(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _forward_logits(self, x):
        """Forward pass returning logits (strip sigmoid for numerical stability)."""
        # Temporarily bypass final sigmoid
        net_without_sigmoid = self.model.net[:-1]   # drop last Sigmoid
        return net_without_sigmoid(x).squeeze(-1)

    def fit(self, features: np.ndarray, labels: np.ndarray,
            val_features: np.ndarray = None, val_labels: np.ndarray = None):
        """Full training loop with early stopping on validation loss."""
        best_val_loss = float('inf')
        patience_count = 0

        for epoch in range(cfg.NODE_MAX_EPOCHS):
            train_loss = self.train_epoch(features, labels)

            val_str = ""
            if val_features is not None:
                val_loss = self._eval_loss(val_features, val_labels)
                val_str  = f"  val_loss={val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss  = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= 10:
                        print(f"  Early stop at epoch {epoch}")
                        break

            if epoch % 10 == 0:
                print(f"  NodeMLP epoch {epoch:3d}  train_loss={train_loss:.4f}{val_str}")

    def _eval_loss(self, features, labels):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.device)
            y = torch.from_numpy(labels).float().to(self.device)
            logits = self._forward_logits(x)
            return self.criterion(logits, y).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
