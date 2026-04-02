"""
Extract bipartite graph features from SCIP's current LP state.
Follows Gasse et al. (NeurIPS 2019) feature set.

Called inside the branching rule callback — SCIP has just solved the LP at
the current node, so all LP solution values are fresh.

Returns:
    constraint_features : np.ndarray  (n_rows, CONSTRAINT_FEAT_DIM)
    edge_indices        : np.ndarray  (2, n_edges)  [row_idx, col_idx]
    edge_features       : np.ndarray  (n_edges, EDGE_FEAT_DIM)
    variable_features   : np.ndarray  (n_cols, VARIABLE_FEAT_DIM)
    candidate_mask      : np.ndarray  (n_cols,) bool — True for LP branch candidates
"""

import numpy as np


# ── Feature dimensions (must match config) ────────────────────────────────────
CONSTRAINT_FEAT_DIM = 5
EDGE_FEAT_DIM       = 1
VARIABLE_FEAT_DIM   = 14   # see _variable_features() below


def _safe_norm(x, eps=1e-8):
    """Normalize array to zero mean, unit std. Safe for constant arrays."""
    mu, sigma = x.mean(), x.std()
    return (x - mu) / (sigma + eps)


def extract_bipartite_graph(model):
    """
    Extract the bipartite (constraint, variable) graph from current SCIP LP state.

    Args:
        model: active pyscipopt.Model (called inside branchexeclp)

    Returns:
        dict with keys: con_feats, edge_index, edge_feats, var_feats, cand_mask
    """
    # ── LP rows (constraints) ─────────────────────────────────────────────────
    rows = model.getLPRowsData()
    n_rows = len(rows)

    # ── LP columns (variables) ────────────────────────────────────────────────
    cols = model.getLPColsData()
    n_cols = len(cols)

    if n_rows == 0 or n_cols == 0:
        return None

    # Map variable name → column index for edge construction
    col_index = {col.getVar().name: idx for idx, col in enumerate(cols)}

    # ── Objective value (used for normalization) ───────────────────────────────
    # Use LP objective value (always available inside branchexeclp), not MIP obj
    try:
        obj_val = abs(model.getLPObjVal())
    except Exception:
        obj_val = 1.0
    if obj_val < 1.0:
        obj_val = 1.0   # prevent division by tiny values inflating features

    # ── Constraint features ───────────────────────────────────────────────────
    con_feats = np.zeros((n_rows, CONSTRAINT_FEAT_DIM), dtype=np.float32)
    n_lps     = max(model.getNLPs(), 1)

    for i, row in enumerate(rows):
        lhs      = row.getLhs()
        rhs      = row.getRhs()
        dualsol  = row.getDualsol()
        row_norm = row.getNorm() if row.getNorm() > 1e-8 else 1.0

        # Compute row activity = sum(coeff * LP_val) + constant
        # Needed for is_tight check. Use try/except in case any col LP val fails.
        activity = row.getConstant()
        try:
            for col_obj, coef in zip(row.getCols(), row.getVals()):
                try:
                    lp_val = col_obj.getPrimsol()
                except AttributeError:
                    try:
                        lp_val = col_obj.getVar().getLPSol()
                    except Exception:
                        lp_val = 0.0
                activity += coef * lp_val
        except Exception:
            activity = 0.0

        # 0: rhs normalized by row norm
        con_feats[i, 0] = rhs / row_norm if not np.isinf(rhs) else 0.0

        # 1: is_tight — constraint is active (binding) at current LP solution
        if not np.isinf(rhs):
            con_feats[i, 1] = float(abs(activity - rhs) < 1e-6)
        elif not np.isinf(lhs):
            con_feats[i, 1] = float(abs(activity - lhs) < 1e-6)
        else:
            # Fallback: use basis status — non-basic row means constraint is tight
            try:
                bstat = row.getBasisStatus()
                con_feats[i, 1] = float(bstat != 'basic')
            except Exception:
                con_feats[i, 1] = 0.0

        # 2: dual solution value (normalized by obj, clipped to [-10, 10])
        con_feats[i, 2] = np.clip(dualsol / obj_val, -10.0, 10.0)

        # 3: LP age (normalized) — defensive
        try:
            con_feats[i, 3] = float(row.getAge()) / n_lps
        except AttributeError:
            con_feats[i, 3] = 0.0

        # 4: objective cosine similarity — defensive
        try:
            con_feats[i, 4] = row.getObjParallelism()
        except AttributeError:
            con_feats[i, 4] = 0.0

    # ── Edges + edge features ─────────────────────────────────────────────────
    edge_rows, edge_cols_idx, edge_vals = [], [], []
    for i, row in enumerate(rows):
        try:
            row_cols = row.getCols()    # list of Column objects
            row_vals = row.getVals()    # list of float coefficients
        except Exception:
            continue

        row_norm = row.getNorm() if row.getNorm() > 1e-8 else 1.0

        for col_obj, coef in zip(row_cols, row_vals):
            try:
                var = col_obj.getVar()
                j   = col_index.get(var.name, None)
            except Exception:
                continue
            if j is not None:
                edge_rows.append(i)
                edge_cols_idx.append(j)
                edge_vals.append(coef / row_norm)   # normalized coefficient

    if len(edge_rows) == 0:
        return None

    edge_index = np.array([edge_rows, edge_cols_idx], dtype=np.int64)
    edge_feats = np.array(edge_vals, dtype=np.float32).reshape(-1, 1)

    # ── Variable features ─────────────────────────────────────────────────────
    var_feats = np.zeros((n_cols, VARIABLE_FEAT_DIM), dtype=np.float32)

    for j, col in enumerate(cols):
        var   = col.getVar()
        vtype = var.vtype()
        lb    = var.getLbLocal()
        ub    = var.getUbLocal()
        obj_c = var.getObj()

        # LP solution value — try column primsol first, fallback to var LP sol
        try:
            lp_val = col.getPrimsol()
        except AttributeError:
            try:
                lp_val = var.getLPSol()
            except Exception:
                lp_val = 0.0

        # Reduced cost
        try:
            rc = col.getRedcost()
        except AttributeError:
            rc = 0.0

        lb = lb if not np.isinf(lb) else 0.0
        ub = ub if not np.isinf(ub) else 1.0

        # Fractionality (only meaningful for integer/binary vars)
        frac = 0.0
        if vtype in ('B', 'I', 'M'):
            frac = lp_val - np.floor(lp_val)
            frac = min(frac, 1.0 - frac)   # symmetric: distance to nearest integer

        # 0-3: variable type one-hot
        var_feats[j, 0] = float(vtype == 'B')
        var_feats[j, 1] = float(vtype == 'I')
        var_feats[j, 2] = float(vtype == 'C')
        var_feats[j, 3] = float(vtype == 'M')   # implicit integer

        # 4: objective coefficient normalized (clipped)
        var_feats[j, 4] = np.clip(obj_c / obj_val, -10.0, 10.0)

        # 5-6: bound indicators
        var_feats[j, 5] = float(not np.isinf(var.getLbLocal()))
        var_feats[j, 6] = float(not np.isinf(var.getUbLocal()))

        # 7-8: at bound
        var_feats[j, 7] = float(abs(lp_val - lb) < 1e-6)
        var_feats[j, 8] = float(abs(lp_val - ub) < 1e-6)

        # 9: solution fractionality
        var_feats[j, 9] = frac

        # 10-12: basis status one-hot — defensive string/int handling
        try:
            bstat = col.getBasisStatus()
            # PySCIPOpt returns string: 'lower', 'basic', 'upper', 'zero'
            if isinstance(bstat, str):
                var_feats[j, 10] = float(bstat == 'lower')
                var_feats[j, 11] = float(bstat == 'basic')
                var_feats[j, 12] = float(bstat == 'upper')
            else:
                # Fallback: encode as-is
                var_feats[j, 11] = 1.0   # assume basic
        except Exception:
            var_feats[j, 11] = 1.0   # default: basic

        # 13: reduced cost normalized (clipped)
        var_feats[j, 13] = np.clip(rc / obj_val, -10.0, 10.0)

    # ── Candidate mask (priority integer/binary candidates only) ──────────────
    # Paper and PySCIPOpt docs: use npriocands, not ncands.
    # nimplcands (implicit integers) should not be branched on in general.
    cand_mask = np.zeros(n_cols, dtype=bool)
    try:
        raw = model.getLPBranchCands()
        cands_all, _, _, ncands, npriocands, _ = raw
        n_prio = npriocands if npriocands > 0 else ncands
        prio_cands = cands_all[:n_prio]
        for var in prio_cands:
            j = col_index.get(var.name, None)
            if j is not None:
                cand_mask[j] = True
    except Exception:
        # Fallback: mark all fractional integer variables
        for j, col in enumerate(cols):
            var = col.getVar()
            if var.vtype() in ('B', 'I'):
                try:
                    lp_val = col.getPrimsol()
                except AttributeError:
                    try:
                        lp_val = var.getLPSol()
                    except Exception:
                        continue
                frac = lp_val - np.floor(lp_val)
                if 1e-6 < frac < 1 - 1e-6:
                    cand_mask[j] = True

    return {
        "con_feats":  con_feats,          # (n_rows, 5)
        "edge_index": edge_index,          # (2, n_edges)
        "edge_feats": edge_feats,          # (n_edges, 1)
        "var_feats":  var_feats,           # (n_cols, 14)
        "cand_mask":  cand_mask,           # (n_cols,) bool
        "n_rows":     n_rows,
        "n_cols":     n_cols,
    }


def get_prenorm_stats(feature_dicts):
    """
    Compute prenorm stats from training feature dicts.
    Also stores sample_graphs so initialize_prenorms can run forward passes
    to collect internal conv-layer aggregation statistics.
    """
    def clean(arr):
        arr = np.array(arr, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.clip(arr, -1e6, 1e6)
        return arr.astype(np.float32)

    all_con = clean(np.concatenate([d["con_feats"] for d in feature_dicts], axis=0))
    all_var = clean(np.concatenate([d["var_feats"] for d in feature_dicts], axis=0))
    all_edg = clean(np.concatenate([d["edge_feats"] for d in feature_dicts], axis=0))

    return {
        "con_mean":      all_con.mean(0).astype(np.float32),
        "con_std":       np.maximum(all_con.std(0), 1e-4).astype(np.float32),
        "var_mean":      all_var.mean(0).astype(np.float32),
        "var_std":       np.maximum(all_var.std(0), 1e-4).astype(np.float32),
        "edg_mean":      all_edg.mean(0).astype(np.float32),
        "edg_std":       np.maximum(all_edg.std(0), 1e-4).astype(np.float32),
        "sample_graphs": feature_dicts,   # passed to initialize_prenorms for conv-layer stats
    }