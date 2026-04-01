"""
Hybrid Search Data Collection — Section 4.2 of Branch Ranking paper.

Procedure per instance:
    At each node t with candidate set X_cand = {x1,...,xd}:

    1. Randomly sample K variables → X_exp = {xe1,...,xek}
    2. For each xe in X_exp:
         a. One-step branch simulation (solve both child LPs)
         b. Roll out full B&B using myopic policy (SB) until termination → R_ei
    3. Select action with highest long-term return (green node in paper Figure 1)
    4. Commit that branch in the real environment
    5. Record:
         D_L : (state, xe_best, R_best)   from long-term exploration
         D_SB: (state, a_SB)              from strong branching during simulation

    Repeat until instance solved.

This is EXPENSIVE — each node requires K full sub-solves.
In practice, limit depth of rollout and use time limits.
"""

import os
import pickle
import numpy as np
from pyscipopt import Model, SCIP_RESULT, Branchrule, Nodesel
from data.feature_extractor import extract_bipartite_graph
from training.reward_assigner import NodeSample
import config as cfg


# ── Strong branching helper ─────────────────────────────────────────────────────

def get_sb_scores(model):
    """
    Get strong branching scores for all LP branch candidates.
    Returns dict: var_name → sb_score

    Uses product scoring: score(x_i) = min(f_down, f_up) * max(f_down, f_up)
    where f_down = frac, f_up = 1-frac. This approximates SB without LP solves.
    For true SB scores, SCIP's vanillafullstrong plugin would be needed.
    """
    try:
        cands, cand_sols, cand_fracs, ncands, *_ = model.getLPBranchCands()
    except Exception:
        return {}

    scores = {}
    for var, sol_val, frac in zip(cands, cand_sols, cand_fracs):
        f_down   = frac
        f_up     = 1.0 - frac
        sb_score = min(f_down, f_up) * max(f_down, f_up)
        scores[var.name] = sb_score

    return scores


def get_current_var_bounds(model):
    """
    Snapshot ALL variable bounds at the current B&B node.

    This is the correct way to reconstruct the branching path for sub-solve replay.
    Each B&B node is fully characterized by its local variable bounds — these ARE the
    accumulated branching decisions from root to this node.

    Returns list of (var_name, 'lb'|'ub', bound_value) for every bound that has been
    tightened from its global (root) value.
    """
    tight_bounds = []
    try:
        for var in model.getVars(transformed=True):
            lb_local  = var.getLbLocal()
            ub_local  = var.getUbLocal()
            lb_global = var.getLbGlobal()
            ub_global = var.getUbGlobal()

            # Only record bounds actually tightened by branching
            if lb_local > lb_global + 1e-8:
                tight_bounds.append((var.name, 'lb', lb_local))
            if ub_local < ub_global - 1e-8:
                tight_bounds.append((var.name, 'ub', ub_local))
    except Exception:
        pass
    return tight_bounds


def solve_subproblem(instance_path, tight_bounds, extra_branch, time_limit=60):
    """
    Solve a restricted sub-B&B from the current node's bound state plus one more branch.

    tight_bounds : list of (var_name, 'lb'|'ub', value)  — current node's bounds
    extra_branch : (var_name, 'lb'|'ub', value)           — one additional branch to test

    Returns:
        n_nodes : int — B&B nodes explored (proxy for trajectory return)
        solved  : bool
    """
    m = Model()
    m.hideOutput(True)
    m.setParam("limits/time", time_limit)
    m.setParam("limits/nodes", 5000)   # cap to avoid runaway sub-solves
    m.readProblem(instance_path)

    # Apply all current-node bounds
    all_bounds = tight_bounds + [extra_branch]
    for var_name, side, bound in all_bounds:
        var = m.getVarByName(var_name)
        if var is None:
            continue
        if side == 'lb':
            m.chgVarLb(var, float(bound))
        else:
            m.chgVarUb(var, float(bound))

    m.optimize()
    return m.getNNodes(), m.getStatus() == 'optimal'


# ── Data recording branchrule ──────────────────────────────────────────────────

class DataCollectionBranchRule(Branchrule):
    """
    Branchrule used during data collection.
    At each node:
        - Extracts graph features
        - Records SB scores (short-term)
        - Optionally runs long-term rollout simulation via bound snapshots

    Key design: sub-solve replay uses get_current_var_bounds() which snapshots
    ALL tightened bounds at the current node. This correctly captures all
    accumulated branching decisions from root without needing to trace history.
    """

    def __init__(self, instance_path, k_explore=None, use_long_term=True):
        super().__init__()
        self.instance_path  = instance_path
        self.k_explore      = k_explore or cfg.K_EXPLORE
        self.use_long_term  = use_long_term

        # Collected data
        self.long_term_groups = []   # list of lists of NodeSample
        self.sb_samples       = []   # list of NodeSample

    def branchexeclp(self, allowaddcons):
        graph = extract_bipartite_graph(self.model)
        if graph is None:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        try:
            cands, cand_sols, cand_fracs, ncands, *_ = self.model.getLPBranchCands()
        except Exception:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if ncands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Build column index map once
        cols        = self.model.getLPColsData()
        col_idx_map = {col.getVar().name: idx for idx, col in enumerate(cols)}

        # Tag graph with node number (used by reward assigner for grouping)
        node_num = -1
        try:
            node_num = self.model.getCurrentNode().getNumber()
        except Exception:
            pass
        graph["node_number"] = node_num

        # Fractionality sum — stored in graph for node selector features
        frac_sum = float(sum(cand_fracs))
        graph["frac_sum"] = frac_sum
        graph["n_cols"]   = len(cols)

        # ── Short-term: record SB scores ──────────────────────────────────────
        sb_scores = get_sb_scores(self.model)
        if sb_scores:
            best_var_name = max(sb_scores, key=sb_scores.get)
            best_col_idx  = col_idx_map.get(best_var_name, 0)
            sb_sample = NodeSample(
                state_graph    = graph,
                action_col_idx = best_col_idx,
                sb_score       = sb_scores[best_var_name]
            )
            sb_sample.is_short_term = True
            sb_sample.reward        = 1.0
            self.sb_samples.append(sb_sample)

        # ── Long-term: sample K variables and roll out ─────────────────────────
        node_group = []
        if self.use_long_term and ncands > 1:
            # Snapshot current node's tightened bounds — replaces _get_current_branch_path()
            # This is the accumulated effect of all branching decisions root→current node.
            current_bounds = get_current_var_bounds(self.model)

            k            = min(self.k_explore, ncands)
            rng          = np.random.RandomState()
            sampled_vars = rng.choice(cands, size=k, replace=False)
            sampled_fracs= {v.name: f for v, f in zip(cands, cand_fracs)}

            for var in sampled_vars:
                frac     = sampled_fracs.get(var.name, 0.5)
                floor_bd = float(int(var.getLbLocal() + frac * (var.getUbLocal() - var.getLbLocal())))
                ceil_bd  = floor_bd + 1.0

                # LEFT branch: add ub = floor_bd
                n_left, _  = solve_subproblem(
                    self.instance_path, current_bounds,
                    extra_branch=(var.name, 'ub', floor_bd), time_limit=60)

                # RIGHT branch: add lb = ceil_bd
                n_right, _ = solve_subproblem(
                    self.instance_path, current_bounds,
                    extra_branch=(var.name, 'lb', ceil_bd), time_limit=60)

                # Return = -min(nodes) → higher return = smaller sub-tree
                traj_return = -float(min(n_left, n_right))

                sample = NodeSample(
                    state_graph       = graph,
                    action_col_idx    = col_idx_map.get(var.name, 0),
                    trajectory_return = traj_return,
                )
                node_group.append(sample)

            if node_group:
                self.long_term_groups.append(node_group)

                # Select best long-term action and commit
                returns  = [s.trajectory_return for s in node_group]
                best_idx = int(np.argmax(returns))
                self.model.branchVar(sampled_vars[best_idx])
                return {"result": SCIP_RESULT.BRANCHED}

        # Default: SCIP handles branching (SB data still recorded above)
        return {"result": SCIP_RESULT.DIDNOTRUN}


# ── Main collection loop ────────────────────────────────────────────────────────

def collect_data_from_instance(instance_path, use_long_term=True, time_limit=None):
    """
    Run one instance through data collection.
    Returns (long_term_groups, sb_samples) for reward assignment.
    """
    time_limit = time_limit or cfg.SCIP_TIME_LIMIT

    model = Model()
    model.hideOutput(True)
    model.setParam("limits/time", time_limit)
    model.readProblem(instance_path)

    # Include our data collection branching rule
    br = DataCollectionBranchRule(instance_path, use_long_term=use_long_term)
    model.includeBranchrule(
        branchrule=br,
        name="data_collection",
        desc="Data collection branching rule",
        priority=1_000_000,
        maxdepth=-1,
        maxbounddist=1.0
    )

    model.optimize()

    return br.long_term_groups, br.sb_samples


def collect_dataset(instance_paths, out_path, use_long_term=True):
    """
    Collect data from multiple instances, save to disk.
    """
    all_long_term_groups = []
    all_sb_samples       = []

    for i, path in enumerate(instance_paths):
        print(f"  Collecting from instance {i+1}/{len(instance_paths)}: {os.path.basename(path)}")
        try:
            lt_groups, sb_samps = collect_data_from_instance(path, use_long_term)
            all_long_term_groups.extend(lt_groups)
            all_sb_samples.extend(sb_samps)
        except Exception as e:
            print(f"    Warning: failed on {path}: {e}")
            continue

    # Save collected data
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'long_term_groups': all_long_term_groups,
            'sb_samples':       all_sb_samples,
        }, f)

    print(f"Saved {len(all_sb_samples)} SB samples, "
          f"{sum(len(g) for g in all_long_term_groups)} long-term samples → {out_path}")

    return all_long_term_groups, all_sb_samples


def load_dataset(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['long_term_groups'], d['sb_samples']
