# RankingBasedBB — Branch Ranking + Neural UCT Node Selection

> **Branch-and-Bound for Mixed-Integer Programming with Offline Reinforcement Learning and Learned Node Selection**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)
[![SCIP](https://img.shields.io/badge/SCIP-10.0.1-green.svg)](https://scipopt.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository implements and extends **Branch Ranking** (Huang et al., ECML-PKDD 2022), an offline reinforcement learning framework for variable selection in branch-and-bound MIP solvers. We additionally contribute a novel **Neural UCT Node Selector** that replaces SCIP's hand-crafted node selection heuristic with a learned MLP using cached GCN embeddings.

**The core research question:** Once variable selection is optimized via offline RL, does learned node selection provide additional improvement?

---

## Background and Motivation

### The Problem: Mixed-Integer Programming

Mixed-Integer Programming (MIP) is a class of optimization problems where some variables must take integer values:

```
min  c^T x
s.t. Ax ≤ b,   x ∈ Z^p × R^(n-p)
```

MIPs appear everywhere — production planning, routing, scheduling, resource allocation. They are NP-hard in general. The standard exact solver is **Branch and Bound (B&B)**, which recursively partitions the solution space into a binary search tree and prunes subtrees using LP relaxation bounds.

Two critical decisions must be made at every B&B iteration:
1. **Variable selection** — which fractional variable to branch on (determines tree *size*)
2. **Node selection** — which open subproblem to expand next (determines traversal *order*)

### Step 1: Gasse et al. — GCN for Imitation Learning (NeurIPS 2019)

[Gasse et al.](https://arxiv.org/abs/1906.01629) made the key observation that a MIP's LP relaxation is naturally a **bipartite graph**: constraint nodes on one side, variable nodes on the other, edges weighted by constraint matrix coefficients. They proposed encoding this as a Graph Convolutional Network (GCN) and training it via **imitation learning** of strong branching — the expensive-but-optimal branching heuristic.

Key innovations:
- No manual feature engineering — the graph structure encodes everything
- **Prenorm layers**: fixed affine normalization initialized from training data, enabling generalization to larger instances than seen during training
- First ML approach to beat SCIP's default branching on a full-fledged solver

### Step 2: Branch Ranking — Offline RL (ECML-PKDD 2022)

[Huang et al.](https://arxiv.org/abs/2207.13701) identified the fundamental limitation of imitation learning: strong branching is **myopic** — it only optimizes the immediate dual bound improvement, ignoring long-term tree structure. A greedy one-step-optimal choice can lead to an exponentially larger tree downstream.

They reformulated variable selection as an **offline RL** problem with three contributions:

**Hybrid Search Data Collection**: At each B&B node, randomly sample K candidate variables. For each, solve complete sub-B&B rollouts to get trajectory returns (negative node count). Select the action with best long-term return and commit it to the real tree.

**Ranking-Based Reward Assignment**: Label state-action pairs as *promising* (reward=1) or not (reward=0) using two criteria:
- **Long-term promising**: trajectory return in top-p% at this node
- **Short-term promising**: highest strong branching score at this node

**Offline Policy Training** (Equation 3 of the paper):
```
max_θ  Σ_{(s,a) ∈ D_hyb}  r(s,a) · log π_θ(a | s)
```
This is weighted cross-entropy — only promising samples contribute gradients.

### Step 3: Our Contribution — Neural UCT Node Selection

Branch Ranking, like all prior work, leaves node selection entirely to SCIP's hand-crafted **hybridestim** heuristic, which blends a fixed-weight estimate score with depth-based plunging. SCIP also provides a **UCT node selector** adapted from game-tree search, but it uses a fixed formula and disables itself after 31 nodes.

**Our hypothesis**: GCN embeddings computed for variable selection already encode rich structural information about each B&B node's LP state. This information, never available to SCIP's heuristics, should improve node selection decisions.

**Neural UCT** replaces SCIP's fixed scoring formula with a learned MLP:

```
score(node_i) = MLP([cached_GCN_embedding(64),   ← structural LP info
                     lowerbound_normalized(1),     ← dual bound quality
                     depth_normalized(1),          ← position in tree
                     frac_sum_normalized(1),        ← LP integrality
                     parent_visits / node_visits(1)]) ← UCT exploration term
```

The visit ratio term preserves UCT's exploration-exploitation balance. The GCN embedding provides information that no hand-crafted heuristic has ever had access to.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SCIP B&B Solver                             │
│                                                                 │
│  branchexeclp() callback          nodeselect() callback         │
│         │                                  │                    │
│         ▼                                  ▼                    │
│  ┌─────────────────┐             ┌──────────────────────┐       │
│  │  LearnedBranch  │             │  NeuralUCTNodeSel    │       │
│  │     Rule        │             │                      │       │
│  │                 │             │  for each open node: │       │
│  │ extract_graph() │             │    emb = cache[node] │       │
│  │      │          │  embedding  │    feats = [emb,lb,  │       │
│  │      ▼          │  cache      │      depth,frac,uct] │       │
│  │  BranchingGCN   │─────────────│    score = MLP(feats)│       │
│  │  (bipartite     │  (shared)   │  return argmax score │       │
│  │   message pass) │             └──────────────────────┘       │
│  │      │          │                        │                   │
│  │  logits(n_cands)│             ┌──────────────────────┐       │
│  │  embeddings     │             │  NodeSelectorMLP     │       │
│  │  (n_cols × 64)  │             │  Linear(68→32)→ReLU  │       │
│  │      │          │             │  Linear(32→16)→ReLU  │       │
│  │  argmax → branch│             │  Linear(16→1)→Sigmoid│       │
│  └─────────────────┘             └──────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Node Selection in SCIP: What Exists and What We Challenge

### The Theoretical Argument for Ignoring Node Selection

The B&B literature has long held that **variable selection dominates node selection** in determining solve time. The argument is theoretically clean:

> Variable selection determines the *structure* of the search tree — how many nodes exist in total. Node selection only determines the *traversal order* of that fixed tree. With infinite time, any traversal order visits the same nodes and reaches the same optimal solution.

With finite time, node selection matters indirectly: expanding a node that leads quickly to a good feasible integer solution tightens the primal bound, which prunes more of the remaining tree before it is explored. But this effect is considered secondary — primal heuristics and SCIP's internal mechanisms already approximate optimal traversal. The empirical literature reinforces this: papers on learning to branch consistently show that improvements to variable selection far outpace any improvements to node selection, and most ML4CO work ignores node selection entirely.

**We challenge this assumption.** Specifically, we hypothesize that when variable selection is already near-optimal (via Branch Ranking), the residual variance in solve time is partially attributable to node selection — and that structural LP information captured in GCN embeddings, which no hand-crafted node selector has ever had access to, can exploit this variance.

---

### Node Selectors Available in SCIP

SCIP exposes a full plugin interface (`Nodesel` class in PySCIPOpt) allowing arbitrary node selection strategies to be registered at runtime. Six node selectors ship with SCIP by default:

| Selector | SCIP name | Brief description |
|---|---|---|
| **Hybrid Estimate** *(default)* | `hybridestim` | Two-phase: plunge (DFS) then global best-estimate. Detailed below. |
| **UCT** | `uct` | Upper Confidence Tree score balancing lower bound and visit counts. Detailed below. |
| **Best First** | `bfs` | Always expand node with globally lowest dual bound |
| **Depth First** | `dfs` | Always expand deepest open node |
| **Best Estimate** | `estimate` | Score = heuristic subtree estimate from pseudocost history |
| **Breadth First** | `breadthfirst` | Level-by-level expansion |
| **Restart DFS** | `restartdfs` | DFS with periodic forced jumps to global best-bound node |

---

### SCIP's Default: `hybridestim`

This is what runs in all experiments unless overridden. It is a carefully engineered two-phase strategy authored by Tobias Achterberg:

**Phase 1 — Plunging (DFS-like):**
When a node is created, SCIP checks whether to "plunge" — continue deeper along the current subtree path rather than returning to the globally best-bound node. Plunging continues while:
- Current plunge depth is within `[minplungedepth, maxplungedepth]`
- The gap ratio `(current_lowerbound - global_lowerbound) / (cutoffbound - global_lowerbound)` is below `maxplungequot` (default 0.25)

Plunging quickly finds feasible integer solutions, which tightens the primal bound and enables aggressive pruning.

**Phase 2 — Global Best-Estimate:**
When plunging stops, the globally best-scored open node is selected:

```
score(node) = (1 - estimweight) · node_estimate  +  estimweight · node_lowerbound
```

where `node_estimate` is a heuristic prediction of the best solution reachable from this node (built from pseudocost history), `node_lowerbound` is the LP relaxation value, and `estimweight = 0.5` by default.

Every `bestnodefreq`-th call, the globally best-bound node is forced to prevent deep-subtree traps.

**Key limitation:** All weights and thresholds are fixed constants tuned on historical SCIP benchmarks. No adaptivity to specific problem structure, and the scoring uses only LP bound values — no graph-structural features.

---

### SCIP's UCT Node Selector (`uct`)

Adapted from game-tree UCB by Sabharwal & Samulowitz (CPAIOR 2011). Rather than scoring all open nodes, it performs a **top-down tree walk** from the root, at each internal node selecting the child `N_i` that maximizes:

```
score(N_i) = -estimate(N_i)  +  weight · visits(parent(N_i)) / visits(N_i)
```

The second term is the **exploration bonus**: nodes visited rarely relative to their parent receive a higher score, preventing indefinite neglect of underexplored subtrees. `visits(n)` is incremented along the entire root-to-selected-leaf path after each expansion.

**Three limitations our Neural UCT addresses:**

| Limitation | SCIP UCT | Our Neural UCT |
|---|---|---|
| Information used | Lower bound + visit counts only | GCN embedding (64-dim structural LP info) + visit counts |
| Scope | Turns off after 31 nodes by default | Active for full solve |
| Adaptivity | `weight = 0.1` fixed constant | MLP learns problem-adaptive exploration-exploitation |

---

### The Experiment: Four-Way Comparison

To empirically test whether node selection matters once variable selection is optimized, we run **four policies on identical problem instances and seeds**:

| Policy | Variable Selection | Node Selection | Purpose |
|---|---|---|---|
| **SCIP Default** | relpcost (heuristic) | hybridestim | Full solver baseline |
| **GCN + hybridestim** | Branch Ranking (learned) | hybridestim | Isolates variable selection gain |
| **GCN + SCIP UCT** | Branch Ranking (learned) | SCIP built-in UCT | Tests existing UCT under learned branching |
| **GCN + Neural UCT** | Branch Ranking (learned) | Learned Neural UCT (ours) | Full proposed system |

The comparison among policies 2, 3, and 4 directly tests node selection in isolation — variable selection is held constant at Branch Ranking, only the node selector changes. This cleanly separates the two contributions.

**The experiment is falsifiable in both directions:**

> **If Neural UCT wins (GO, win rate ≥ 60%, p < 0.05):** The theoretical argument is incomplete. Structural LP information in GCN embeddings captures node quality signals that dual bounds alone cannot. Learned node selection provides significant improvement even after variable selection is optimized. This would be the first demonstration of ML-driven node selection outperforming SCIP's state-of-the-art heuristic under identical branching conditions.

> **If Neural UCT does not win (NO GO):** We provide the first systematic empirical validation of the implicit assumption in the learning-to-branch literature — that once variable selection is optimized, node selection becomes a second-order effect regardless of the information available to the selector. This is also a publishable finding that justifies the community's focus on variable selection alone.

The SCIP UCT baseline (policy 3) further allows us to isolate whether any improvement comes from the UCT *structure* (exploration-exploitation balance) or from the learned MLP *content* (GCN embeddings).

All four policies use identical SCIP settings following the paper: cuts at root node only, no restarts, same randomization seeds.

---

## Repository Structure

```
RankingBasedBB_4NodeNVariable_Selection/
│
├── config.py                    # All hyperparameters (single source of truth)
├── train.py                     # Main training pipeline orchestration
├── evaluate.py                  # Evaluation + GO/NO GO statistical test
├── smoke_test.py                # 25-check test suite (no SCIP needed)
│
├── data/
│   ├── instance_generator.py    # 4 benchmark MIP classes (set cover, auction, facility, indset)
│   └── feature_extractor.py     # Bipartite graph features from SCIP LP state
│
├── models/
│   ├── gcn.py                   # Bipartite GCN with prenorm layers (Gasse et al. architecture)
│   └── node_mlp.py              # Neural UCT node selector MLP + trainer
│
├── training/
│   ├── data_collector.py        # Hybrid search data collection (Section 4.2 of paper)
│   ├── reward_assigner.py       # Ranking-based reward assignment (Section 4.3)
│   └── trainer.py               # GCN + NodeMLP training loops
│
├── branching/
│   └── branch_rule.py           # PySCIPOpt Branchrule plugin (variable selection)
│
├── node_selection/
│   └── node_selector.py         # PySCIPOpt Nodesel plugin (Neural UCT node selection)
│
└── utils/
    ├── embedding_cache.py        # Shared GCN embedding cache between branch rule + node selector
    └── metrics.py                # Evaluation metrics, GO/NO GO test
```

---

## Installation

**Requirements:** Python 3.10+, SCIP 10.0.1, CUDA 12.4 (optional)

```bash
# 1. Clone
git clone https://github.com/Zeleckto/RankingBasedBB_4NodeNVariable_Selection.git
cd RankingBasedBB_4NodeNVariable_Selection

# 2. Virtual environment
python -m venv venv
source venv/Scripts/activate    # Windows Git Bash
# source venv/bin/activate      # Linux/Mac

# 3. PyTorch with CUDA 12.4
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Remaining dependencies
pip install -r requirements.txt

# 5. Verify SCIP is found
python -c "import pyscipopt; m = pyscipopt.Model(); print('SCIP OK:', m.version())"

# 6. Smoke test
python smoke_test.py
```

---

## Usage

### Quick Start (reduced scale)

```bash
# Edit config.py first for fast testing:
# K_EXPLORE = 3, SCIP_TIME_LIMIT = 120

# Step 1: Generate instances
python train.py --problem setcover --skip-collect --skip-gcn --skip-node

# Step 2: Collect offline data
python train.py --problem setcover --skip-gcn --skip-node

# Step 3: Train GCN (Branch Ranking)
python train.py --problem setcover --skip-collect

# Step 4: Collect node selection training data
python evaluate.py --problem setcover --collect-node-data

# Step 5: Train NodeMLP (Neural UCT)
python train.py --problem setcover --skip-generate --skip-collect --skip-gcn

# Step 6: Evaluate
python evaluate.py --problem setcover --difficulty easy --n-instances 20 --n-seeds 5
```

### Full Experiment (paper scale)

```bash
# config.py: K_EXPLORE = 30, SCIP_TIME_LIMIT = 3600
for p in setcover auction facility indset; do
    python train.py --problem $p
    python evaluate.py --problem $p --collect-node-data
    python train.py --problem $p --skip-generate --skip-collect --skip-gcn
    python evaluate.py --problem $p --difficulty easy   --n-instances 20
    python evaluate.py --problem $p --difficulty medium --n-instances 20
    python evaluate.py --problem $p --difficulty hard   --n-instances 20
done
```

### Switching Node Selection Mode

```python
# config.py
NODE_SEL_MODE = "default"     # SCIP hybridestim (baseline)
NODE_SEL_MODE = "neural_uct"  # Our Neural UCT (proposed)
```

---

## Key Hyperparameters

| Parameter | Paper Value | Default | Description |
|---|---|---|---|
| `K_EXPLORE` | 30 | 30 | Sub-solves per node during data collection |
| `TOP_P` | 0.10 | 0.10 | Top-p% threshold for long-term promising label |
| `SB_PROPORTION` (h) | 0.70–0.95 | 0.70 | Fraction of short-term samples in training dataset |
| `GCN_LR` | — | 1e-3 | GCN Adam learning rate |
| `NODE_POS_WEIGHT` | — | 10.0 | BCE positive class weight for NodeMLP |
| `JOINT_LAMBDA` | — | 0.1 | Weight of node loss in joint training |

---

## Benchmarks

Following Gasse et al. and Branch Ranking, we evaluate on four NP-hard MIP benchmark classes:

| Problem | Easy | Medium | Hard |
|---|---|---|---|
| **Set Covering** | 500 rows, 1000 cols | 1000 rows | 2000 rows |
| **Combinatorial Auction** | 100 items, 500 bids | 200 items, 1000 bids | 300 items, 1500 bids |
| **Capacitated Facility Location** | 100 customers, 100 facilities | 200 customers | 400 customers |
| **Maximum Independent Set** | 500-node graph | 1000-node graph | 1500-node graph |

Models are trained on **easy** instances only and evaluated on easy/medium/hard to test generalization.

---

## Results

> **Note**: Results pending full experimental run. Placeholders below.

### Variable Selection Performance (Branch Ranking vs Baselines)

*Table: 1-shifted geometric mean of solve times (seconds). Lower is better. Results on Set Covering.*

| Method | Easy Time | Medium Time | Hard Time | Easy Nodes | Medium Nodes | Hard Nodes |
|---|---|---|---|---|---|---|
| SCIP Default (Relpcost) | — | — | — | — | — | — |
| GCN (Gasse et al.) | — | — | — | — | — | — |
| **Branch Ranking (Ours)** | — | — | — | — | — | — |

<!-- Insert result table image here -->
<!-- ![Set Covering Results](results/setcover_table.png) -->

### Node Selection: Neural UCT vs Baselines

*Table: Solve time SGM (seconds) comparing three policies. Lower is better.*

| Policy | Branch Rule | Node Selector | Easy | Medium | Hard |
|---|---|---|---|---|---|
| SCIP Baseline | relpcost | hybridestim | — | — | — |
| GCN Only | Branch Ranking | hybridestim | — | — | — |
| **GCN + Neural UCT** | Branch Ranking | Neural UCT (ours) | — | — | — |

<!-- Insert Neural UCT comparison figure here -->
<!-- ![Node Selection Comparison](results/neural_uct_comparison.png) -->

### GO/NO GO Statistical Test Results

*Wilcoxon signed-rank test: does Neural UCT beat GCN+SCIP-node on ≥60% of instances with p<0.05?*

| Benchmark | Difficulty | Win Rate | p-value | Decision |
|---|---|---|---|---|
| Set Covering | Easy | — | — | — |
| Set Covering | Medium | — | — | — |
| Set Covering | Hard | — | — | — |
| Combinatorial Auction | Easy | — | — | — |
| Facility Location | Easy | — | — | — |
| Max Independent Set | Easy | — | — | — |

<!-- Insert GO/NO GO summary figure here -->
<!-- ![GO/NO GO Results](results/go_no_go.png) -->

### Convergence Curves

<!-- Insert training loss curves here -->
<!-- ![GCN Training](results/gcn_training_curves.png) -->
<!-- ![NodeMLP Training](results/node_mlp_training_curves.png) -->

---

## Technical Contributions

### 1. Branch Ranking Implementation

Complete faithful implementation of Huang et al. (ECML-PKDD 2022):

- **Hybrid search data collection** using variable bounds snapshot approach: at each B&B node, all accumulated branching decisions are captured by snapshotting locally-tightened variable bounds (`lb_local > lb_global`), which fully characterizes the current node without path-history tracking
- **Ranking-based reward assignment**: long-term promising (top-10% trajectory returns per node), short-term promising (best fractionality-proxy SB score), mixed with problem-specific ratio h
- **GCN architecture** with prenorm layers initialized from training data statistics (enabling generalization across instance scales)
- **Correct paper settings**: cuts at root node only, no restarts, randomization seeds for reproducibility

### 2. Neural UCT Node Selector (Novel)

A learned replacement for SCIP's fixed-formula node selection heuristic:

- **GCN embedding cache**: when the branch rule processes a node, it mean-pools the 64-dim variable embeddings and stores them keyed by SCIP node number. The node selector retrieves these at selection time — zero additional GCN forward passes
- **68-dim input features**: 64-dim structural embedding + normalized lower bound + depth + fractionality sum + UCT visit ratio
- **UCT exploration term preserved**: `parent_visits / (node_visits + 1)` allows the model to balance exploitation (expand structurally promising nodes) with exploration (avoid ignoring undervisited subtrees)
- **Binary cross-entropy training** with pos_weight=10 to handle ~2-10% positive class rate
- **SolverBundle**: single object holding shared EmbeddingCache between branch rule and node selector, ensuring embeddings flow correctly between the two SCIP plugins

### Bug Fixes vs Previous Learning-to-Branch Code

This implementation corrects several issues present in naive reimplementations:

1. **npriocands vs ncands**: SCIP's `getLPBranchCands()` returns both priority candidates and implicit integers; only priority candidates should be branched on
2. **Correct floor/ceil bounds**: `floor_bd = floor(lp_sol_value)` not `lb + frac*(ub-lb)`
3. **Exact local index mapping**: using `np.where(cand_positions == col_idx)` not `searchsorted` for mapping column indices to GCN logit positions

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{huang2022branch,
  title     = {Branch Ranking for Efficient Mixed-Integer Programming via Offline Ranking-Based Policy Learning},
  author    = {Huang, Zeren and Chen, Wenhao and Zhang, Weinan and Shi, Chuhan and Liu, Furui and Zhen, Hui-Ling and Yuan, Mingxuan and Hao, Jianye and Yu, Yong and Wang, Jun},
  booktitle = {Machine Learning and Knowledge Discovery in Databases (ECML-PKDD)},
  year      = {2022}
}

@inproceedings{gasse2019exact,
  title     = {Exact Combinatorial Optimization with Graph Convolutional Neural Networks},
  author    = {Gasse, Maxime and Ch{\'e}telat, Didier and Ferroni, Nicola and Charlin, Laurent and Lodi, Andrea},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
```

---

## References

1. Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). **Exact Combinatorial Optimization with Graph Convolutional Neural Networks.** *NeurIPS 2019.* [[paper]](https://arxiv.org/abs/1906.01629) [[code]](https://github.com/ds4dm/learn2branch)

2. Huang, Z., Chen, W., Zhang, W., Shi, C., Liu, F., Zhen, H.-L., Yuan, M., Hao, J., Yu, Y., & Wang, J. (2022). **Branch Ranking for Efficient Mixed-Integer Programming via Offline Ranking-Based Policy Learning.** *ECML-PKDD 2022.* [[paper]](https://arxiv.org/abs/2207.13701)

3. Sabharwal, A., & Samulowitz, H. (2011). Guiding Combinatorial Optimization with UCT. *CPAIOR 2011.* (Motivation for UCT node selection in MIP)

4. Khalil, E. B., Le Bodic, P., Song, L., Nemhauser, G., & Dilkina, B. (2016). Learning to Branch in Mixed Integer Programming. *AAAI 2016.*

---

*IIT Delhi 2026*