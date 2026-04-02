"""
Main training script — runs the full Branch Ranking pipeline.

Steps:
    1. Generate instances (if not already done)
    2. Collect offline data via hybrid search
    3. Assign ranking-based rewards
    4. Initialize prenorm layers from data stats
    5. Train GCN (Branch Ranking)
    6. Collect node selection training data
    7. Train NodeMLP
    8. (Optional) Joint fine-tuning

Usage:
    python train.py [--problem setcover] [--skip-collect] [--skip-gcn] [--skip-node]
"""

import os
import argparse
import shutil
import torch
import numpy as np
import pickle

import config as cfg
from data.instance_generator import generate_instances
from training.data_collector import collect_dataset, load_dataset
from training.reward_assigner import (
    assign_long_term_rewards, assign_short_term_rewards,
    build_training_dataset, build_node_training_labels
)
from training.trainer import GCNTrainer, initialize_prenorms
from models.gcn import build_gcn
from models.node_mlp import NodeSelectorMLP, NodeMLPTrainer
from utils.embedding_cache import EmbeddingCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--problem",       default=cfg.PROBLEM_TYPE, choices=cfg.PROBLEM_TYPES)
    p.add_argument("--skip-generate", action="store_true", help="Skip instance generation")
    p.add_argument("--skip-collect",  action="store_true", help="Skip data collection")
    p.add_argument("--skip-gcn",      action="store_true", help="Skip GCN training")
    p.add_argument("--skip-node",     action="store_true", help="Skip NodeMLP training")
    p.add_argument("--device",        default=cfg.DEVICE)
    return p.parse_args()


def main():
    args   = parse_args()
    pt     = args.problem
    device = args.device

    os.makedirs(cfg.INSTANCE_DIR,   exist_ok=True)
    os.makedirs(cfg.DATA_DIR,       exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Generate instances ─────────────────────────────────────────────
    if not args.skip_generate:
        print(f"\n[1/5] Generating {pt} instances...")
        instance_dir = os.path.join(cfg.INSTANCE_DIR, pt, cfg.TRAIN_SIZE)
        train_paths  = generate_instances(pt, cfg.TRAIN_SIZE,
                                          n_instances=200,
                                          out_dir=instance_dir)
        val_dir   = os.path.join(cfg.INSTANCE_DIR, pt, "val")
        val_paths = generate_instances(pt, cfg.TRAIN_SIZE,
                                        n_instances=50,
                                        out_dir=val_dir,
                                        seed=cfg.SEED + 1)
        print(f"  Generated {len(train_paths)} train, {len(val_paths)} val instances")
    else:
        instance_dir = os.path.join(cfg.INSTANCE_DIR, pt, cfg.TRAIN_SIZE)
        val_dir      = os.path.join(cfg.INSTANCE_DIR, pt, "val")
        train_paths  = sorted([os.path.join(instance_dir, f)
                        for f in os.listdir(instance_dir) if f.endswith('.lp')]) \
                       if os.path.exists(instance_dir) else []
        val_paths    = sorted([os.path.join(val_dir, f)
                        for f in os.listdir(val_dir) if f.endswith('.lp')]) \
                       if os.path.exists(val_dir) else []
        print(f"  Found {len(train_paths)} train, {len(val_paths)} val instances")

    # Early exit: if only generating instances, nothing else to do
    if args.skip_collect and args.skip_gcn and args.skip_node:
        print("\nInstance generation complete.")
        return

    # ── Step 2: Collect offline data ───────────────────────────────────────────
    data_path = os.path.join(cfg.DATA_DIR, f"{pt}_data.pkl")
    if not args.skip_collect:
        print(f"\n[2/5] Collecting offline data (hybrid search)...")
        print(f"  K_EXPLORE={cfg.K_EXPLORE}, SCIP_TIME_LIMIT={cfg.SCIP_TIME_LIMIT}s")
        print(f"  Using {min(50, len(train_paths))} instances for collection")
        collect_dataset(train_paths[:50], data_path, use_long_term=True)
    else:
        if not os.path.exists(data_path):
            print(f"\nERROR: --skip-collect specified but {data_path} does not exist.")
            print("  Run without --skip-collect first to generate the data.")
            return
        print(f"  Skipping data collection, loading from {data_path}")

    # ── Step 3: Reward assignment ──────────────────────────────────────────────
    # Only runs if GCN or NodeMLP training is requested
    if not args.skip_gcn or not args.skip_node:
        print(f"\n[3/5] Assigning rewards...")
        lt_groups, sb_samples = load_dataset(data_path)

        lt_flat    = assign_long_term_rewards(lt_groups, top_p=cfg.TOP_P)
        sb_samples = assign_short_term_rewards(sb_samples)

        h = {"setcover": 0.70, "auction": 0.90, "facility": 0.95, "indset": 0.90}.get(pt, cfg.SB_PROPORTION)
        combined_samples, train_graphs, train_rewards = build_training_dataset(lt_flat, sb_samples, h=h)

        n_lt = sum(s.is_long_term  for s in combined_samples)
        n_sb = sum(s.is_short_term for s in combined_samples)
        print(f"  Training samples: {len(train_graphs)}  (LT: {n_lt}, SB: {n_sb})")
        if len(train_graphs) == 0:
            print("  WARNING: 0 training samples. Data collection may have failed.")
            print("  Check that instances are solvable and K_EXPLORE > 0.")
            return

    # ── Step 4: Train GCN ──────────────────────────────────────────────────────
    gcn_path = os.path.join(cfg.CHECKPOINT_DIR, f"{pt}_gcn_best.pt")

    if not args.skip_gcn:
        print(f"\n[4/5] Training GCN on {device}...")
        gcn = build_gcn(cfg)
        initialize_prenorms(gcn, train_graphs)

        trainer = GCNTrainer(gcn, device=device)

        split      = int(0.9 * len(train_graphs))
        train_data = (combined_samples[:split], train_graphs[:split], train_rewards[:split])
        val_data   = (combined_samples[split:], train_graphs[split:], train_rewards[split:])

        trainer.fit(train_data, val_data, checkpoint_dir=cfg.CHECKPOINT_DIR)
        # Save with problem-specific name
        best = os.path.join(cfg.CHECKPOINT_DIR, "gcn_best.pt")
        if os.path.exists(best):
            shutil.copy(best, gcn_path)
            print(f"  Saved GCN checkpoint → {gcn_path}")
    else:
        print(f"  Skipping GCN training")

    # ── Step 5: Train NodeMLP ──────────────────────────────────────────────────
    node_mlp_path = os.path.join(cfg.CHECKPOINT_DIR, f"{pt}_node_mlp.pt")

    if not args.skip_node:
        print(f"\n[5/5] Training NodeSelectorMLP on {device}...")

        node_data_path = os.path.join(cfg.DATA_DIR, f"{pt}_node_data.pkl")
        if not os.path.exists(node_data_path):
            print(f"  Node data not found at {node_data_path}")
            print("  Run: python evaluate.py --collect-node-data first")
            print("  Skipping NodeMLP training.")
            return

        with open(node_data_path, 'rb') as f:
            node_feats, node_labels = pickle.load(f)
        print(f"  Loaded {len(node_feats)} node samples  "
              f"(positive rate: {node_labels.mean():.2%})")

        if node_labels.mean() == 0:
            print("  WARNING: 0% positive rate. Labeling may be broken.")
            print("  Check evaluate.py --collect-node-data output.")
            return

        node_mlp = NodeSelectorMLP()
        trainer  = NodeMLPTrainer(node_mlp, device=device)

        split = int(0.9 * len(node_feats))
        trainer.fit(
            features=node_feats[:split],
            labels=node_labels[:split],
            val_features=node_feats[split:],
            val_labels=node_labels[split:],
            checkpoint_path=node_mlp_path
        )
        trainer.save(node_mlp_path)
        print(f"  Saved NodeMLP → {node_mlp_path}")
    else:
        print(f"  Skipping NodeMLP training")

    print(f"\nTraining complete. Checkpoints in {cfg.CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()