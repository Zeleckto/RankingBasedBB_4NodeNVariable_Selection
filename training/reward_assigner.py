"""
Ranking-based Reward Assignment — Section 4.3 of Branch Ranking paper.

Given the offline dataset D_hyb = D_L ∪ D_SB:

D_L  = {(state, action, trajectory_return)} from long-term rollouts
D_SB = {(state, action)}                    from strong branching (short-term)

Reward assignment:
    r(s,a) = 1  if (s,a) is Long-Term Promising  OR Short-Term Promising
    r(s,a) = 0  otherwise

Long-Term Promising  (Definition 1):
    Trajectory return R(s,a) ranks in top-p% of all trajectories starting from s.
    In practice: among K explored actions at a node, top-p% by return get r=1.

Short-Term Promising (Definition 2):
    Action a has the highest SB score (dual bound improvement) at state s.
    Exactly 1 action per node gets r=1 from this criterion.

Final training dataset:
    Proportion h = short-term samples, (1-h) = long-term samples.
    h is a tunable hyperparameter (paper uses 0.7–0.95 per problem class).

The training objective (Equation 3):
    max_θ  Σ_{(s,a) ~ D_hyb} r(s,a) * log π_θ(a|s)

This is equivalent to weighted cross-entropy where reward = sample weight.
"""

import numpy as np
from typing import List, Dict, Tuple
import config as cfg


# ── Data structures ────────────────────────────────────────────────────────────

class NodeSample:
    """One branching decision observation."""
    __slots__ = ['state_graph', 'action_col_idx', 'trajectory_return',
                 'sb_score', 'is_long_term', 'is_short_term', 'reward']

    def __init__(self, state_graph, action_col_idx,
                 trajectory_return=None, sb_score=None):
        self.state_graph        = state_graph       # dict from feature_extractor
        self.action_col_idx     = action_col_idx    # int: column index of chosen variable
        self.trajectory_return  = trajectory_return # float: negative node count (higher=better)
        self.sb_score           = sb_score          # float or None
        self.is_long_term       = False
        self.is_short_term      = False
        self.reward             = 0.0


# ── Core reward assignment ──────────────────────────────────────────────────────

def assign_long_term_rewards(node_groups: List[List[NodeSample]],
                              top_p: float = None) -> List[NodeSample]:
    """
    For each group of samples from the SAME node (different actions explored),
    label top-p% by trajectory_return as long-term promising.

    node_groups: list of lists, each inner list = all K explored actions at one node
    Returns: flat list of NodeSample with is_long_term set
    """
    top_p = top_p or cfg.TOP_P
    flat  = []

    for group in node_groups:
        # Filter to samples that have a trajectory return
        valid = [s for s in group if s.trajectory_return is not None]
        if not valid:
            flat.extend(group)
            continue

        returns = np.array([s.trajectory_return for s in valid])
        # Higher return = smaller search tree = better
        threshold = np.percentile(returns, 100 * (1 - top_p))

        for s in valid:
            if s.trajectory_return >= threshold:
                s.is_long_term = True
                s.reward = 1.0

        flat.extend(group)

    return flat


def assign_short_term_rewards(samples: List[NodeSample]) -> List[NodeSample]:
    """
    Among samples from the SAME node, mark the one with highest SB score
    as short-term promising.

    Groups samples by their state identity (node number) then applies per-group.
    """
    # Group by node (state) — use id(state_graph) as proxy
    from collections import defaultdict
    groups = defaultdict(list)
    for s in samples:
        # Use graph identity; in practice, group by node number stored in state_graph
        node_id = s.state_graph.get("node_number", id(s.state_graph))
        groups[node_id].append(s)

    for node_id, group in groups.items():
        sb_samples = [s for s in group if s.sb_score is not None]
        if not sb_samples:
            continue
        best = max(sb_samples, key=lambda s: s.sb_score)
        best.is_short_term = True
        best.reward = 1.0

    return samples


def build_training_dataset(long_term_samples: List[NodeSample],
                             sb_samples: List[NodeSample],
                             h: float = None) -> Tuple[List, List, List]:
    """
    Combine long-term and short-term samples into final training dataset.

    h: proportion of short-term samples in final dataset (per paper, h ∈ [0.7, 0.95])

    Returns:
        training_samples  : list of NodeSample with reward set
        graphs            : list of state_graph dicts
        rewards           : list of float (0 or 1)
    """
    h = h if h is not None else cfg.SB_PROPORTION

    # Only keep promising samples (reward=1)
    lt_promising = [s for s in long_term_samples if s.is_long_term]
    sb_promising = [s for s in sb_samples        if s.is_short_term]

    # Balance according to h
    n_sb_target = int(len(lt_promising) * h / (1 - h + 1e-8)) if h < 1.0 else len(sb_promising)
    n_sb_target = min(n_sb_target, len(sb_promising))

    # Sample short-term to hit proportion h
    rng = np.random.RandomState(cfg.SEED)
    sb_selected = rng.choice(sb_promising, size=n_sb_target, replace=False).tolist() \
                  if n_sb_target < len(sb_promising) else sb_promising

    combined = lt_promising + sb_selected
    rng.shuffle(combined)

    graphs  = [s.state_graph for s in combined]
    rewards = [s.reward      for s in combined]

    return combined, graphs, rewards


def compute_weighted_ce_loss(logits_list, target_indices, rewards):
    """
    Weighted cross-entropy: L = -Σ r(s,a) * log π(a|s)
    where π(a|s) = softmax(logits)[a]

    logits_list   : list of tensors, each (n_cands_i,) for node i
    target_indices: list of int, the chosen candidate's local index in logits
    rewards       : list of float, reward for each sample

    Returns: scalar tensor
    """
    import torch
    import torch.nn.functional as F

    total_loss = torch.tensor(0.0)
    count = 0

    for logits, target_idx, reward in zip(logits_list, target_indices, rewards):
        if reward == 0.0:
            continue  # zero-reward samples don't contribute to loss
        log_probs = F.log_softmax(logits, dim=0)
        loss = -reward * log_probs[target_idx]
        total_loss = total_loss + loss
        count += 1

    return total_loss / max(count, 1)


# ── Node MLP training data construction ────────────────────────────────────────

def build_node_training_labels(trajectories: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    From collected B&B trajectories, generate labels for node MLP training.

    Each trajectory dict contains:
        'node_features' : list of (features_array, node_number) per node selection
        'optimal_path'  : set of node_numbers on path to best solution

    Returns:
        features : (N, node_input_dim) array
        labels   : (N,) binary array
    """
    all_features = []
    all_labels   = []

    for traj in trajectories:
        for feats, node_num in traj.get('node_features', []):
            all_features.append(feats)
            label = 1 if node_num in traj.get('optimal_path', set()) else 0
            all_labels.append(label)

    if not all_features:
        return np.zeros((0, cfg.NODE_INPUT_DIM)), np.zeros(0, dtype=np.int32)

    return (np.stack(all_features, axis=0).astype(np.float32),
            np.array(all_labels, dtype=np.float32))
