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

def get_sb_scores(model, prio_cands=None, prio_fracs=None):
    """
    Get strong branching scores for priority LP branch candidates.

    Uses product scoring: score(x_i) = min(f, 1-f)^2
    where f = fractionality of x_i (distance to nearest integer).
    This matches the dual-bound improvement proxy common in the literature.

    Paper uses true SB (solve 2 LPs per candidate). That requires
    SCIP's vanillafullstrong which is slow. This proxy is fast and
    correctly ranks highly-fractional (near-0.5) variables highest.

    Args:
        prio_cands: list of Variables (priority candidates). If None, fetched.
        prio_fracs: list of fracs matching prio_cands order. If None, fetched.
    """
    if prio_cands is None or prio_fracs is None:
        try:
            cands, _, fracs, ncands, npriocands, _ = model.getLPBranchCands()
            n = npriocands if npriocands > 0 else ncands
            prio_cands = cands[:n]
            prio_fracs = fracs[:n]
        except Exception:
            return {}

    scores = {}
    for var, frac in zip(prio_cands, prio_fracs):
        # Symmetric fractionality: distance to nearest integer
        sym_frac = min(frac, 1.0 - frac)
        scores[var.name] = sym_frac * sym_frac   # score ∈ [0, 0.25], max at frac=0.5
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
            # Paper uses npriocands (priority candidates) not ncands.
            # nimplcands are implicit integers — should not be branched on in general.
            cands, cand_sols, cand_fracs, ncands, npriocands, nimplcands = \
                self.model.getLPBranchCands()
        except Exception:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Use npriocands: actual integer/binary priority candidates only
        n_prio = npriocands if npriocands > 0 else ncands
        if n_prio == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Slice to priority candidates only (they are sorted first)
        prio_cands = cands[:n_prio]
        prio_sols  = cand_sols[:n_prio]
        prio_fracs = cand_fracs[:n_prio]

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

        # frac_sum: sum of fractionalities of PRIORITY candidates only
        # frac = distance to nearest integer for each candidate
        frac_sum = float(sum(min(f, 1.0 - f) for f in prio_fracs))
        graph["frac_sum"] = frac_sum
        graph["n_cols"]   = len(cols)

        # ── Short-term: record SB scores at current node ───────────────────────
        # Note: Paper's D_SB is collected from SB decisions INSIDE the K rollout
        # simulations. Here we record the best SB candidate at the current node
        # as an approximation (rollout SB would require intercepting sub-SCIP runs).
        sb_scores = get_sb_scores(self.model, prio_cands, prio_fracs)
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
        if self.use_long_term and n_prio > 1:
            # Snapshot current node's tightened bounds.
            # These ARE all accumulated branching decisions from root → current node.
            current_bounds = get_current_var_bounds(self.model)

            k            = min(self.k_explore, n_prio)
            rng          = np.random.RandomState()
            sample_idx   = rng.choice(n_prio, size=k, replace=False)
            sampled_vars = [prio_cands[i] for i in sample_idx]
            sampled_sols = [prio_sols[i]  for i in sample_idx]

            for var, lp_sol in zip(sampled_vars, sampled_sols):
                # Correct floor/ceil: branch at the LP solution value
                # floor_bd = floor(x*), ceil_bd = ceil(x*)
                floor_bd = float(np.floor(lp_sol))
                ceil_bd  = float(np.ceil(lp_sol))

                if floor_bd >= ceil_bd - 1e-8:
                    # LP solution is already nearly integer, skip
                    continue

                # Per paper Figure 1: each variable gets ONE trajectory return R_ei
                # from rolling out the simulation tree T_ei. We run both children
                # and take the return from the better child (paper does not specify
                # which child to follow; using min nodes = best return is conservative).
                n_left, _  = solve_subproblem(
                    self.instance_path, current_bounds,
                    extra_branch=(var.name, 'ub', floor_bd), time_limit=60)

                n_right, _ = solve_subproblem(
                    self.instance_path, current_bounds,
                    extra_branch=(var.name, 'lb', ceil_bd), time_limit=60)

                # R_ei = -nodes explored. Higher = smaller tree = better.
                # Use min(n_left, n_right): best reachable outcome from branching on x_e.
                traj_return = -float(min(n_left, n_right))

                col_idx = col_idx_map.get(var.name, None)
                if col_idx is None:
                    continue

                sample = NodeSample(
                    state_graph       = graph,
                    action_col_idx    = col_idx,
                    trajectory_return = traj_return,
                )
                node_group.append(sample)

            if node_group:
                self.long_term_groups.append(node_group)

                # Select best long-term action (green node in paper Figure 1) and commit
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
    SCIP settings match the paper (Section 5.1):
        - Cutting planes at root node only
        - No restarts
        - All other SCIP parameters default
    Returns (long_term_groups, sb_samples) for reward assignment.
    """
    time_limit = time_limit or cfg.SCIP_TIME_LIMIT

    model = Model()
    model.hideOutput(True)
    model.setParam("limits/time", time_limit)

    # ── Paper Section 5.1 exact SCIP settings ─────────────────────────────────
    # "cutting planes enabled at the root node"
    model.setParam("separating/maxroundsroot", -1)   # unlimited cuts at root
    model.setParam("separating/maxrounds",      0)   # no cuts after root
    # "deactivate solver restarts" (following Gasse et al.)
    model.setParam("presolving/maxrestarts", 0)

    model.readProblem(instance_path)

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