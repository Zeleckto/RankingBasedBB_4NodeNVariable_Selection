"""
Generate benchmark MIP instances following Gasse et al. (NeurIPS 2019).
Four classes: Set Covering, Combinatorial Auction, Capacitated Facility Location,
Maximum Independent Set.

All instances saved as .lp files for SCIP to read.
"""

import os
import numpy as np
from pyscipopt import Model, quicksum


def generate_setcover(n_rows, n_cols, density=0.05, rng=None, filepath=None):
    """
    Set covering: min c^T x s.t. Ax >= 1, x in {0,1}^n
    Following Balas & Ho (1980): each column covers a random subset of rows.
    n_rows: number of rows (constraints)
    n_cols: number of columns (variables)
    """
    rng = rng or np.random.RandomState()
    
    # Ensure every row is covered by at least one column
    A = rng.binomial(1, density, (n_rows, n_cols)).astype(float)
    for row in range(n_rows):
        if A[row].sum() == 0:
            A[row, rng.randint(n_cols)] = 1.0
    
    c = rng.randint(1, 100, n_cols).astype(float)
    
    model = Model()
    model.hideOutput(True)
    
    x = [model.addVar(vtype='B', name=f'x{j}', obj=c[j]) for j in range(n_cols)]
    
    for i in range(n_rows):
        model.addCons(quicksum(A[i, j] * x[j] for j in range(n_cols) if A[i, j] > 0) >= 1.0,
                      name=f'row{i}')
    
    model.setMinimize()
    
    if filepath:
        model.writeProblem(filepath)
    return model


def generate_auction(n_items, n_bids, rng=None, filepath=None):
    """
    Combinatorial Auction: max sum(price_i * y_i) s.t. each item in at most one bid.
    Following Leyton-Brown et al. arbitrary relationships procedure.
    """
    rng = rng or np.random.RandomState()
    
    # Generate bids: each bid covers a random bundle of items
    bids = []
    for _ in range(n_bids):
        bundle_size = rng.randint(1, min(5, n_items) + 1)
        bundle = rng.choice(n_items, bundle_size, replace=False).tolist()
        # Price: additive with synergy bonus
        price = sum(rng.uniform(0, 1) for _ in bundle) * (1 + 0.2 * rng.random())
        bids.append((bundle, price))
    
    model = Model()
    model.hideOutput(True)
    
    # Variables: y_i = 1 if bid i is accepted
    y = [model.addVar(vtype='B', name=f'y{i}', obj=-bids[i][1]) for i in range(n_bids)]
    
    # Constraints: each item sold at most once
    for item in range(n_items):
        covering_bids = [i for i, (bundle, _) in enumerate(bids) if item in bundle]
        if covering_bids:
            model.addCons(quicksum(y[i] for i in covering_bids) <= 1.0, name=f'item{item}')
    
    model.setMinimize()  # minimizing negative revenue = maximizing revenue
    
    if filepath:
        model.writeProblem(filepath)
    return model


def generate_facility(n_customers, n_facilities, rng=None, filepath=None):
    """
    Capacitated Facility Location: minimize fixed + transport costs.
    Following Cornuejols et al. (1991).
    """
    rng = rng or np.random.RandomState()
    
    # Random costs
    transport = rng.uniform(0, 1, (n_customers, n_facilities))
    fixed = rng.uniform(0, 1, n_facilities) * 0.5
    demands = rng.uniform(1, 3, n_customers)
    capacities = rng.uniform(3, 6, n_facilities)
    
    model = Model()
    model.hideOutput(True)
    
    # y[j] = 1 if facility j is opened
    y = [model.addVar(vtype='B', name=f'y{j}', obj=fixed[j]) for j in range(n_facilities)]
    # x[i,j] = fraction of customer i served by facility j
    x = [[model.addVar(vtype='C', lb=0.0, ub=1.0, name=f'x{i}_{j}',
                       obj=transport[i, j])
          for j in range(n_facilities)] for i in range(n_customers)]
    
    # Each customer fully served
    for i in range(n_customers):
        model.addCons(quicksum(x[i][j] for j in range(n_facilities)) >= 1.0,
                      name=f'cust{i}')
    
    # Capacity constraints
    for j in range(n_facilities):
        model.addCons(
            quicksum(demands[i] * x[i][j] for i in range(n_customers)) <= capacities[j] * y[j],
            name=f'cap{j}')
    
    model.setMinimize()
    
    if filepath:
        model.writeProblem(filepath)
    return model


def generate_indset(n_nodes, affinity=4, rng=None, filepath=None):
    """
    Maximum Independent Set on Erdos-Renyi random graph.
    Following Bergman et al., affinity=4 (expected degree).
    """
    rng = rng or np.random.RandomState()
    
    p_edge = affinity / n_nodes
    
    # Generate edges
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p_edge:
                edges.append((i, j))
    
    model = Model()
    model.hideOutput(True)
    
    # x[i] = 1 if node i in independent set; maximize sum(x)
    x = [model.addVar(vtype='B', name=f'x{i}', obj=-1.0) for i in range(n_nodes)]
    
    # Edge constraints: at most one endpoint in independent set
    for i, j in edges:
        model.addCons(x[i] + x[j] <= 1.0, name=f'edge{i}_{j}')
    
    model.setMinimize()  # minimizing -sum(x) = maximizing sum(x)
    
    if filepath:
        model.writeProblem(filepath)
    return model


# ── Batch generation ───────────────────────────────────────────────────────────

GENERATORS = {
    "setcover": generate_setcover,
    "auction":  generate_auction,
    "facility": generate_facility,
    "indset":   generate_indset,
}

SIZE_ARGS = {
    # (problem_type, difficulty) → kwargs for generator
    ("setcover", "easy"):   dict(n_rows=500,  n_cols=1000),
    ("setcover", "medium"): dict(n_rows=1000, n_cols=1000),
    ("setcover", "hard"):   dict(n_rows=2000, n_cols=1000),
    ("auction",  "easy"):   dict(n_items=100, n_bids=500),
    ("auction",  "medium"): dict(n_items=200, n_bids=1000),
    ("auction",  "hard"):   dict(n_items=300, n_bids=1500),
    ("facility", "easy"):   dict(n_customers=100, n_facilities=100),
    ("facility", "medium"): dict(n_customers=200, n_facilities=100),
    ("facility", "hard"):   dict(n_customers=400, n_facilities=100),
    ("indset",   "easy"):   dict(n_nodes=500),
    ("indset",   "medium"): dict(n_nodes=1000),
    ("indset",   "hard"):   dict(n_nodes=1500),
}


def generate_instances(problem_type, difficulty, n_instances, out_dir, seed=42):
    """
    Generate n_instances and save as .lp files to out_dir.
    Returns list of filepaths.
    """
    os.makedirs(out_dir, exist_ok=True)
    gen_fn = GENERATORS[problem_type]
    kwargs = SIZE_ARGS[(problem_type, difficulty)].copy()
    
    filepaths = []
    rng = np.random.RandomState(seed)
    
    for i in range(n_instances):
        fp = os.path.join(out_dir, f"{problem_type}_{difficulty}_{i:05d}.lp")
        if not os.path.exists(fp):
            gen_fn(**kwargs, rng=rng, filepath=fp)
        filepaths.append(fp)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_instances} {problem_type} {difficulty} instances")
    
    return filepaths


if __name__ == "__main__":
    import config as cfg
    
    for pt in cfg.PROBLEM_TYPES:
        for diff, n in [("easy", 100), ("medium", 20), ("hard", 20)]:
            out = os.path.join(cfg.INSTANCE_DIR, pt, diff)
            print(f"Generating {pt} {diff}...")
            generate_instances(pt, diff, n, out)
    print("Done.")
