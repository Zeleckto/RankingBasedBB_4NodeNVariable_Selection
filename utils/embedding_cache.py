"""
Embedding Cache — stores mean-pooled GCN variable embeddings + scalar node stats per B&B node.

Lifecycle:
    Branching rule (branchexeclp):
        1. Runs GCN → gets var_embeddings (n_cols, 64)
        2. Computes node_emb = mean(var_embeddings) → (64,)
        3. Calls cache.store(node_number, node_emb, frac_sum, n_cols)

    Node selector (nodeselect):
        1. For each open node, calls cache.get(node_number)
        2. Uses (embedding, frac_sum, n_cols) as part of MLP input features

Memory management:
    - Cache is a simple dict: node_number → (np.ndarray, float, int)
    - Call cache.prune(active_node_numbers) periodically to free pruned nodes
    - Default values (zeros, 0.0, 1) returned for uncached nodes
"""

import numpy as np
import threading
from typing import Optional, Set, Tuple
import config as cfg


class EmbeddingCache:
    def __init__(self, emb_dim: int = None):
        self.emb_dim   = emb_dim or cfg.EMBEDDING_DIM
        # node_number → (embedding, frac_sum, n_cols)
        self._cache    = {}
        self._lock     = threading.Lock()
        self._default_emb = np.zeros(self.emb_dim, dtype=np.float32)
        self._n_hits   = 0
        self._n_misses = 0

    # ── Write ──────────────────────────────────────────────────────────────────

    def store(self, node_number: int, embedding: np.ndarray,
              frac_sum: float = 0.0, n_cols: int = 1):
        """Store embedding + scalar stats for a node."""
        assert embedding.shape == (self.emb_dim,), \
            f"Expected ({self.emb_dim},), got {embedding.shape}"
        with self._lock:
            self._cache[node_number] = (
                embedding.astype(np.float32),
                float(frac_sum),
                int(n_cols)
            )

    def store_batch(self, node_numbers, embeddings: np.ndarray,
                    frac_sums=None, n_cols_list=None):
        """Store multiple entries at once."""
        with self._lock:
            for k, (num, emb) in enumerate(zip(node_numbers, embeddings)):
                frac = frac_sums[k] if frac_sums is not None else 0.0
                nc   = n_cols_list[k] if n_cols_list is not None else 1
                self._cache[int(num)] = (emb.astype(np.float32), float(frac), int(nc))

    # ── Read ───────────────────────────────────────────────────────────────────

    def get(self, node_number: int) -> Tuple[np.ndarray, float, int]:
        """
        Retrieve (embedding, frac_sum, n_cols).
        Returns (zeros, 0.0, 1) if not cached.
        """
        with self._lock:
            entry = self._cache.get(node_number, None)
            if entry is None:
                self._n_misses += 1
                return self._default_emb.copy(), 0.0, 1
            self._n_hits += 1
            return entry

    def get_embedding(self, node_number: int) -> np.ndarray:
        """Retrieve embedding only."""
        emb, _, _ = self.get(node_number)
        return emb

    def get_batch(self, node_numbers) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve embeddings + scalars for multiple nodes.
        Returns (embeddings, frac_sums, n_cols) arrays.
        """
        results = [self.get(n) for n in node_numbers]
        embs     = np.stack([r[0] for r in results], axis=0)
        frac_sums = np.array([r[1] for r in results], dtype=np.float32)
        n_cols    = np.array([r[2] for r in results], dtype=np.int32)
        return embs, frac_sums, n_cols

    def has(self, node_number: int) -> bool:
        with self._lock:
            return node_number in self._cache

    # ── Memory management ──────────────────────────────────────────────────────

    def prune(self, active_node_numbers: Set[int]):
        with self._lock:
            dead = [k for k in self._cache if k not in active_node_numbers]
            for k in dead:
                del self._cache[k]

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._n_hits   = 0
            self._n_misses = 0

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._n_hits + self._n_misses
        return self._n_hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "cached_nodes": self.size,
            "hit_rate":     f"{self.hit_rate:.2%}",
            "hits":         self._n_hits,
            "misses":       self._n_misses,
        }

    def __repr__(self):
        return f"EmbeddingCache(size={self.size}, hit_rate={self.hit_rate:.2%})"
