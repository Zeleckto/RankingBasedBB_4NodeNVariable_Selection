"""
Central config for Branch Ranking MIP solver.
All hyperparameters live here — change nothing elsewhere.
"""

# ── Problem / Benchmark ────────────────────────────────────────────────────────
PROBLEM_TYPES   = ["setcover", "auction", "facility", "indset"]
PROBLEM_TYPE    = "setcover"          # active benchmark

# ── Instance sizes ─────────────────────────────────────────────────────────────
INSTANCE_SIZES = {
    "setcover":  {"easy": (500, 1000),  "medium": (1000, 1000), "hard": (2000, 1000)},
    "auction":   {"easy": (100, 500),   "medium": (200, 1000),  "hard": (300, 1500)},
    "facility":  {"easy": (100, 100),   "medium": (200, 100),   "hard": (400, 100)},
    "indset":    {"easy": (500,),       "medium": (1000,),      "hard": (1500,)},
}
TRAIN_SIZE      = "easy"             # train on easy only, eval on all
N_TRAIN         = 10_000             # instances generated for training
N_VAL           = 2_000
N_TEST_EACH     = 20                 # per difficulty level

# ── Data Collection (Hybrid Search) ───────────────────────────────────────────

K_EXPLORE       = 30        # number of variables sampled per node for long-term search
TOP_P           = 0.10      # top-p% trajectories labelled long-term promising
SB_PROPORTION   = 0.70      # h: fraction of short-term (SB) samples in final dataset
N_TRAIN_SAMPLES = 50_000    # state-action pairs for GCN training
N_VAL_SAMPLES   = 5_000
SCIP_TIME_LIMIT = 3600      # seconds per instance during collection

#Just verification runs:  



# ── GCN Architecture ──────────────────────────────────────────────────────────
CONSTRAINT_FEAT_DIM = 5     # per-constraint features
EDGE_FEAT_DIM       = 1     # per-edge features
VARIABLE_FEAT_DIM   = 14    # per-variable features (see feature_extractor.py)
EMBEDDING_DIM       = 64    # hidden dimension throughout GCN
GCN_LAYERS          = 1     # number of bipartite conv rounds (paper uses 1)

# ── GCN Training ──────────────────────────────────────────────────────────────
GCN_LR              = 1e-3
GCN_LR_DECAY        = 0.2           # multiply lr when plateau
GCN_PATIENCE        = 10            # epochs before lr decay
GCN_STOP_PATIENCE   = 20            # epochs before early stop
GCN_BATCH_SIZE      = 32
GCN_MAX_EPOCHS      = 1000
GCN_WEIGHT_DECAY    = 0.0



# ── Node MLP Architecture ─────────────────────────────────────────────────────
NODE_INPUT_DIM      = EMBEDDING_DIM + 4   # embedding + [lb_norm, depth_norm, frac_norm, visit_ratio]
NODE_HIDDEN_DIMS    = [32, 16]
NODE_OUTPUT_DIM     = 1

# ── Node MLP Training ─────────────────────────────────────────────────────────
NODE_LR             = 1e-3
NODE_WEIGHT_DECAY   = 1e-4
NODE_BATCH_SIZE     = 64
NODE_MAX_EPOCHS     = 50
NODE_POS_WEIGHT     = 10.0          # BCE positive class weight (optimal nodes are rare)
JOINT_LAMBDA        = 0.1           # weight for node loss in joint training



# ── Node Selector ─────────────────────────────────────────────────────────────
NODE_SEL_MODE       = "default"     # "default" | "neural_uct" | "attention"
UCT_WEIGHT          = 0.1           # exploration weight in UCT score
UCT_NODE_LIMIT      = -1            # -1 = never turn off neural UCT (unlike SCIP's 31)

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_SEEDS          = 5             # number of SCIP seeds per instance
EVAL_TIME_LIMIT     = 3600          # seconds

# ── Paths ──────────────────────────────────────────────────────────────────────
INSTANCE_DIR        = "instances"
DATA_DIR            = "collected_data"
CHECKPOINT_DIR      = "checkpoints"
RESULTS_DIR         = "results"

# ── Misc ───────────────────────────────────────────────────────────────────────
import torch as _torch
DEVICE              = "cuda" if _torch.cuda.is_available() else "cpu"
SEED                = 42
