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

*IIT Delhi — MCD412 BTP2 — 2024-25*
