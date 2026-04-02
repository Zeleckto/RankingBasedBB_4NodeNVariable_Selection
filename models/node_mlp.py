"""
Node Selector MLP — scores open B&B nodes using cached GCN embeddings + scalar features.

Input features per node (68-dim):
    cached_embedding   (64)  : mean-pooled GCN variable embeddings from node creation
    lowerbound_norm    ( 1)  : (lb - root_lb) / (cutoff - root_lb)
    depth_norm         ( 1)  : depth / max_depth_seen
    frac_sum_norm      ( 1)  : sum of fractionalities / n_cols
    visit_ratio        ( 1)  : parent_visits / (node_visits + 1)  <- UCT exploration term

Output: scalar in [0,1] (probability that this node is on optimal path)

Training: Binary Cross-Entropy with pos_weight=10 for class imbalance.
"""

import os
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
        return self.net(x).squeeze(-1)

    def score_nodes(self, node_features: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(node_features).float()
            return self.forward(x).numpy()


def build_node_features(node_embedding, lowerbound, root_lowerbound, cutoff,
                        depth, max_depth, frac_sum, n_cols,
                        node_visits, parent_visits) -> np.ndarray:
    emb     = node_embedding
    span    = cutoff - root_lowerbound
    lb_norm = float(np.clip((lowerbound - root_lowerbound) / (span + 1e-8), 0.0, 1.0)) \
              if span > 1e-8 else 0.0
    depth_norm  = float(depth) / float(max(max_depth, 1))
    frac_norm   = frac_sum / float(max(n_cols, 1))
    visit_ratio = float(parent_visits) / float(node_visits + 1)
    scalars = np.array([lb_norm, depth_norm, frac_norm, visit_ratio], dtype=np.float32)
    return np.concatenate([emb, scalars])


class NodeMLPTrainer:
    def __init__(self, model: NodeSelectorMLP, device='cpu'):
        self.model     = model.to(device)
        self.device    = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.NODE_LR, weight_decay=cfg.NODE_WEIGHT_DECAY)
        pos_weight     = torch.tensor([cfg.NODE_POS_WEIGHT], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _forward_logits(self, x):
        return self.model.net[:-1](x).squeeze(-1)

    def train_epoch(self, features, labels) -> float:
        self.model.train()
        idx = np.random.permutation(len(features))
        total, n = 0.0, 0
        for start in range(0, len(features), cfg.NODE_BATCH_SIZE):
            bi = idx[start : start + cfg.NODE_BATCH_SIZE]
            x  = torch.from_numpy(features[bi]).float().to(self.device)
            y  = torch.from_numpy(labels[bi]).float().to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self._forward_logits(x), y)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            n     += 1
        return total / max(n, 1)

    def _eval_loss(self, features, labels) -> float:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.device)
            y = torch.from_numpy(labels).float().to(self.device)
            return self.criterion(self._forward_logits(x), y).item()

    def fit(self, features, labels, val_features=None, val_labels=None,
            checkpoint_path=None):
        best_val   = float('inf')
        patience   = 0
        start      = 0

        if checkpoint_path:
            latest = checkpoint_path.replace('.pt', '_latest.pt')
            resume = latest if os.path.exists(latest) else \
                     checkpoint_path if os.path.exists(checkpoint_path) else None
            if resume:
                try:
                    ckpt = torch.load(resume, map_location=self.device)
                    if isinstance(ckpt, dict) and 'model_state' in ckpt:
                        self.model.load_state_dict(ckpt['model_state'])
                        self.optimizer.load_state_dict(ckpt['opt_state'])
                        best_val = ckpt.get('val_loss', float('inf'))
                        start    = ckpt.get('epoch',    0) + 1
                        patience = ckpt.get('patience', 0)
                    else:
                        self.model.load_state_dict(ckpt)
                    print(f"  NodeMLP resumed from epoch {start}")
                except Exception as e:
                    print(f"  NodeMLP resume failed ({e}), starting fresh")

        for epoch in range(start, cfg.NODE_MAX_EPOCHS):
            train_loss = self.train_epoch(features, labels)
            val_str    = ""

            if val_features is not None:
                val_loss = self._eval_loss(val_features, val_labels)
                val_str  = f"  val_loss={val_loss:.4f}"
                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    if checkpoint_path:
                        self._save_full(checkpoint_path, epoch, patience, best_val)
                else:
                    patience += 1
                    if patience >= 10:
                        print(f"  Early stop at epoch {epoch}")
                        break

            if checkpoint_path and epoch % 10 == 0:
                self._save_full(checkpoint_path.replace('.pt', '_latest.pt'),
                                epoch, patience, best_val)

            if epoch % 10 == 0:
                print(f"  NodeMLP epoch {epoch:3d}  train_loss={train_loss:.4f}{val_str}")

    def _save_full(self, path, epoch, patience, val_loss):
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        torch.save({'model_state': self.model.state_dict(),
                    'opt_state':   self.optimizer.state_dict(),
                    'epoch':       epoch,
                    'patience':    patience,
                    'val_loss':    val_loss}, path)

    def save(self, path):
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            self.model.load_state_dict(ckpt['model_state'])
        else:
            self.model.load_state_dict(ckpt)