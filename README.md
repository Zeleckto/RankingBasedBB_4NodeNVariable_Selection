# Branch Ranking MIP — Neural B&B Solver

Implementation of **Branch Ranking** (Huang et al., ECML-PKDD 2022) with an added
**Neural UCT Node Selector** as a novel contribution on top.

## Architecture

```
branch_ranking_mip/
├── config.py                    # All hyperparameters
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation + GO/NO GO test
├── data/
│   ├── instance_generator.py    # 4 benchmark MIP classes
│   └── feature_extractor.py     # Bipartite graph features from SCIP
├── models/
│   ├── gcn.py                   # Bipartite GCN (Gasse et al. architecture)
│   └── node_mlp.py              # Node selector MLP (our addition)
├── training/
│   ├── data_collector.py        # Hybrid search data collection
│   ├── reward_assigner.py       # Ranking-based reward (top-p% + SB)
│   └── trainer.py               # GCN + NodeMLP training loops
├── branching/
│   └── branch_rule.py           # PySCIPOpt Branchrule plugin
├── node_selection/
│   └── node_selector.py         # Neural UCT Nodesel plugin ← our contribution
└── utils/
    ├── embedding_cache.py        # Shared GCN embedding cache
    └── metrics.py                # Solve time, nodes, GO/NO GO
```

## Installation (Windows + Git Bash)

```bash
# 1. Install SCIP: download from https://scipopt.org/index.php#download
#    Set SCIP_DIR environment variable to installation path

# 2. Create venv
python -m venv venv
source venv/Scripts/activate   # Git Bash

# 3. Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Generate instances + collect data + train everything
python train.py --problem setcover

# Skip slow steps during development
python train.py --problem setcover --skip-collect --skip-gcn

# Evaluate
python evaluate.py --problem setcover --difficulty easy

# Collect node selection training data (needed for NodeMLP)
python evaluate.py --collect-node-data
```

## Training Pipeline

1. **Generate instances** — Set covering, Auction, Facility Location, Indep. Set
2. **Hybrid search collection** — K=30 rollouts per node, long-term + SB samples
3. **Reward assignment** — top-p% long-term + best SB = reward 1, rest = 0
4. **Train GCN** — weighted cross-entropy: `L = -Σ r(s,a) log π(a|s)`
5. **Collect node data** — run GCN solver, log node selection events + labels
6. **Train NodeMLP** — BCE: `L = -Σ [y log σ(f(x)) + (1-y) log(1-σ(f(x)))]`
7. **Evaluate** — GO/NO GO: win rate ≥ 60% + Wilcoxon p < 0.05

## Novel Contribution: Neural UCT Node Selector

Replaces SCIP's fixed-formula hybridestim with a learned MLP score:

```
score(node) = MLP([cached_GCN_embedding(64),
                   lowerbound_norm(1),
                   depth_norm(1),
                   frac_sum_norm(1),
                   parent_visits / node_visits(1)])   ← UCT exploration term
```

- **GCN embedding**: cached when node is created during branching (free reuse)
- **UCT term**: preserves exploration-exploitation balance from SCIP's UCT selector
- **Learned**: trained offline from Branch Ranking trajectories with BCE loss
- **Scope**: active for entire solve (unlike SCIP UCT which stops at 31 nodes)

## Citation

```
@inproceedings{huang2022branch,
  title={Branch Ranking for Efficient Mixed-Integer Programming via Offline Ranking-Based Policy Learning},
  author={Huang, Zeren and Chen, Wenhao and Zhang, Weinan and others},
  booktitle={ECML-PKDD},
  year={2022}
}

@inproceedings{gasse2019exact,
  title={Exact Combinatorial Optimization with Graph Convolutional Neural Networks},
  author={Gasse, Maxime and Ch{\'e}telat, Didier and Ferroni, Nicola and Charlin, Laurent and Lodi, Andrea},
  booktitle={NeurIPS},
  year={2019}
}
```
