"""
Evaluation metrics for MIP solving.
Standard B&B benchmarking metrics following Gasse et al.
"""

import numpy as np
import time
from pyscipopt import Model
from typing import List, Dict, Optional


def shifted_geometric_mean(values, shift=1.0):
    """
    1-shifted geometric mean — standard MIP benchmark metric.
    Avoids issues with very small/zero values.
    """
    values = np.array(values, dtype=float)
    return np.exp(np.mean(np.log(values + shift))) - shift


def solve_instance(instance_path, branchrule=None, nodesel=None,
                   time_limit=3600, seed=0, verbose=False,
                   paper_settings=True):
    """
    Solve one instance and return metrics.

    paper_settings=True applies the exact SCIP settings from the paper (Section 5.1):
        - Cuts at root node only
        - No restarts
        - randomization seeds for reproducibility
    """
    model = Model()
    if not verbose:
        model.hideOutput(True)

    model.setParam("limits/time", time_limit)
    model.setParam("randomization/permutationseed", seed)
    model.setParam("randomization/lpseed", seed)

    if paper_settings:
        # Paper: "cutting planes enabled at the root node"
        model.setParam("separating/maxroundsroot", -1)
        model.setParam("separating/maxrounds",      0)
        # Paper: "deactivate solver restarts" (following Gasse et al. Appendix)
        model.setParam("presolving/maxrestarts", 0)

    model.readProblem(instance_path)

    if branchrule is not None:
        model.includeBranchrule(
            branchrule=branchrule,
            name="learned_br",
            desc="Learned branching rule",
            priority=1_000_000,
            maxdepth=-1,
            maxbounddist=1.0
        )

    if nodesel is not None:
        model.includeNodesel(
            nodesel=nodesel,
            name="learned_ns",
            desc="Learned node selector",
            stdpriority=1_000_000,
            memsavepriority=0
        )

    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0

    status = model.getStatus()
    solved = status == 'optimal'
    n_nodes = model.getNNodes() if solved else None

    primal = model.getPrimalbound()
    dual   = model.getDualbound()
    gap    = abs(primal - dual) / (abs(primal) + 1e-8) if solved else float('inf')

    return {
        "solve_time":       min(elapsed, time_limit),
        "n_nodes":          n_nodes,
        "status":           status,
        "obj_val":          primal,
        "primal_dual_gap":  gap,
        "solved":           solved,
    }


def evaluate_policy(instance_paths, branchrule_factory, nodesel_factory=None,
                    n_seeds=5, time_limit=3600, verbose=False):
    """
    Evaluate a branching policy over multiple instances and seeds.

    branchrule_factory: callable() → new Branchrule instance per solve
    nodesel_factory   : callable() → new Nodesel instance per solve (or None)

    Returns dict with aggregate metrics.
    """
    results = []

    for path in instance_paths:
        for seed in range(n_seeds):
            br = branchrule_factory() if branchrule_factory else None
            ns = nodesel_factory()    if nodesel_factory    else None

            r = solve_instance(path, br, ns, time_limit, seed, verbose)
            r['instance'] = path
            r['seed']     = seed
            results.append(r)

    return aggregate_results(results, time_limit)


def aggregate_results(results: List[Dict], time_limit=3600):
    """Compute aggregate statistics from a list of solve results."""
    times   = [r['solve_time'] for r in results]
    solved  = [r for r in results if r['solved']]
    nodes   = [r['n_nodes'] for r in solved if r['n_nodes'] is not None]
    gaps    = [r['primal_dual_gap'] for r in results]

    agg = {
        "n_total":          len(results),
        "n_solved":         len(solved),
        "solve_rate":       len(solved) / max(len(results), 1),
        "time_mean":        np.mean(times),
        "time_sgm":         shifted_geometric_mean(times),          # primary metric
        "time_std":         np.std(times),
        "nodes_mean":       np.mean(nodes) if nodes else None,
        "nodes_sgm":        shifted_geometric_mean(nodes) if nodes else None,
        "gap_mean":         np.mean(gaps),
        "times_all":        times,
        "nodes_all":        nodes,
    }
    return agg


def print_results(name, agg):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Solved:      {agg['n_solved']}/{agg['n_total']} ({agg['solve_rate']:.1%})")
    print(f"  Time (SGM):  {agg['time_sgm']:.2f}s")
    print(f"  Time (mean): {agg['time_mean']:.2f}s  ± {agg['time_std']:.2f}s")
    if agg['nodes_sgm']:
        print(f"  Nodes (SGM): {agg['nodes_sgm']:.0f}")
    print(f"  Gap (mean):  {agg['gap_mean']:.4f}")


def compare_policies(baseline_agg, proposed_agg, alpha=0.05):
    """
    GO/NO GO test: does proposed beat baseline on ≥60% of instances?
    Uses paired comparison on solve times.
    Returns dict with win_rate and wilcoxon p-value.
    """
    from scipy import stats

    b_times = np.array(baseline_agg['times_all'])
    p_times = np.array(proposed_agg['times_all'])

    n = min(len(b_times), len(p_times))
    b_times = b_times[:n]
    p_times = p_times[:n]

    wins = (p_times < b_times).sum()
    win_rate = wins / n

    # Wilcoxon signed-rank test
    try:
        stat, pvalue = stats.wilcoxon(b_times, p_times, alternative='greater')
    except Exception:
        pvalue = 1.0

    go = win_rate >= 0.60 and pvalue < alpha

    print(f"\n  GO/NO GO Analysis")
    print(f"  Win rate:  {win_rate:.1%}  (need ≥60%)")
    print(f"  p-value:   {pvalue:.4f}   (need <{alpha})")
    print(f"  Decision:  {'✓ GO' if go else '✗ NO GO'}")

    return {"win_rate": win_rate, "p_value": pvalue, "go": go}