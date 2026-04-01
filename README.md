# RankingBasedBB: Branch Ranking + Neural UCT Node Selection

A research-oriented implementation of **learning-based Branch-and-Bound (B&B)** for Mixed Integer Programming (MIP), combining:

* 📌 **Branch Ranking (ECML-PKDD 2022)**
* 📌 **GCN-based variable selection (Gasse et al., NeurIPS 2019)**
* 🆕 **Neural UCT Node Selection (novel contribution)**

---

## 🚀 Overview

This project implements a **learning-to-branch pipeline** where:

1. A **Graph Convolutional Network (GCN)** learns to select branching variables.
2. A **Node Selection MLP** learns to guide tree traversal using a **Neural UCT strategy**.
3. Training is performed via **offline reinforcement learning** using hybrid search.

---

## 🧠 Key Contributions

### ✅ Implemented (from literature)

* **MDP formulation of branching**
* **Hybrid search data collection**
* **Ranking-based reward assignment**
* **Bipartite GCN architecture**
* **Offline RL training objective**
* **Benchmarks (4 problem classes)**

### 🆕 Novel Contribution

* **Neural UCT Node Selector**

  * Replaces SCIP’s fixed node selection heuristic
  * Uses **GCN embeddings + UCT exploration term**
  * Learns adaptive exploration/exploitation strategy

---

## 📂 Project Structure

```
.
├── config.py                  # Global configuration
├── train.py                  # Main training pipeline
├── evaluate.py               # Evaluation + GO/NO-GO test
├── smoke_test.py             # Sanity checks
│
├── data/
│   ├── instance_generator.py # Generate MIP instances
│   ├── feature_extractor.py  # Graph features from SCIP
│
├── models/
│   ├── gcn.py                # Branching GCN
│   ├── node_mlp.py           # Node selection MLP
│
├── training/
│   ├── data_collector.py     # Hybrid search data collection
│   ├── reward_assigner.py    # Reward labeling
│   ├── trainer.py            # Training loops
│
├── branching/
│   └── branch_rule.py        # Learned branching rule
│
├── node_selection/
│   └── node_selector.py      # Neural UCT selector
│
├── utils/
│   ├── embedding_cache.py    # Shared embeddings
│   └── metrics.py            # Evaluation metrics
│
├── instances/                # Generated MIP instances
├── collected_data/           # Offline datasets
├── checkpoints/              # Model weights
```

---

## ⚙️ Setup

### 1. Create and activate virtual environment

```bash
python -m venv venv
source venv/Scripts/activate   # Git Bash (Windows)
```

### 2. Install dependencies

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Install SCIP + PySCIPOpt

Download SCIP:
👉 https://scipopt.org/index.php#download

Then verify:

```bash
python -c "import pyscipopt; m = pyscipopt.Model(); print('SCIP OK')"
```

---

## 🧪 Quick Test

```bash
python smoke_test.py
```

Runs ~25 checks:

* GCN forward pass
* embedding cache
* reward assignment
* NodeMLP pipeline

---

## 🏋️ Full Training Pipeline

### Step 1 — Generate Instances

```bash
python train.py --problem setcover --skip-collect --skip-gcn --skip-node
```

Output:

```
instances/{problem}/{train,val}/
```

---

### Step 2 — Collect Offline Data (Slow ⚠️)

```bash
python train.py --problem setcover --skip-gcn --skip-node
```

Output:

```
collected_data/setcover_data.pkl
```

---

### Step 3 — Train GCN (Branch Ranking)

```bash
python train.py --problem setcover --skip-collect
```

Output:

```
checkpoints/setcover_gcn_best.pt
```

---

### Step 4 — Collect Node Data

```bash
python evaluate.py --problem setcover --collect-node-data
```

Output:

```
collected_data/setcover_node_data.pkl
```

---

### Step 5 — Train Node Selector

```bash
python train.py --problem setcover --skip-generate --skip-collect --skip-gcn
```

Output:

```
checkpoints/setcover_node_mlp.pt
```

---

## 📊 Evaluation

```bash
python evaluate.py --problem setcover --difficulty easy --n-instances 20
```

### Benchmarks

* Set Cover
* Combinatorial Auction
* Facility Location
* Independent Set

---

## 🧮 Model Details

### 🔹 Branching Model (GCN)

* Bipartite graph: constraints ↔ variables
* Features:

  * Constraint: 5-dim
  * Variable: 14-dim
  * Edge: 1-dim
* Uses:

  * Prenorm layers (critical for generalization)
  * Sum aggregation (not mean)

---

### 🔹 Node Selector (Neural UCT)

Input (68-dim):

* 64-dim GCN embedding
* Lower bound
* Depth
* Fractionality
* UCT visit ratio

Architecture:

```
68 → 32 → 16 → 1 (sigmoid)
```

---

## 🔄 Data Flow (During Solve)

1. SCIP calls **branch rule**
2. GCN predicts best variable
3. Embeddings cached
4. Node selector scores all open nodes
5. Best node chosen via MLP + UCT

---

## ⚡ Key Config Parameters

Edit in `config.py`:

| Parameter         | Description                          |
| ----------------- | ------------------------------------ |
| `K_EXPLORE`       | Rollouts per node (speed vs quality) |
| `NODE_SEL_MODE`   | `"default"` or `"neural_uct"`        |
| `SCIP_TIME_LIMIT` | Max solve time                       |
| `TOP_P`           | Long-term reward threshold           |
| `SB_PROPORTION`   | Short-term data ratio                |

---

## 🧪 GO / NO-GO Test

Compare:

| Policy   | Branching | Node Selection |
| -------- | --------- | -------------- |
| Baseline | SCIP      | SCIP           |
| GCN      | Learned   | SCIP           |
| Proposed | Learned   | Neural UCT     |

### Success Criteria

* ✅ ≥ 60% win rate
* ✅ Wilcoxon p-value < 0.05

---

## ⚠️ Known Limitations

* Strong branching uses proxy (not SCIP native)
* Node labels are simplified (can be improved)
* Full optimal-path labeling not implemented yet

---

## 📌 Future Work

* Integrate SCIP strong branching API
* Improve node labeling via optimal path tracking
* Joint training (GCN + NodeMLP)
* Scale to larger MIP instances

---

## 📖 References

* Huang et al., *Branch Ranking for MIP*, ECML-PKDD 2022
* Gasse et al., *Learning to Branch*, NeurIPS 2019

---

## 👤 Author

Developed by **Zeleckto**

---

## ⭐ If you find this useful

Consider starring the repo and citing the work!
