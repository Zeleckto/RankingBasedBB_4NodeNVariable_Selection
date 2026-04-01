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

    # ── Step 1: Generate instances ─────────────────────────────────────────────
    if not args.skip_generate:
        print(f"\n[1/5] Generating {pt} instances...")
        instance_dir = os.path.join(cfg.INSTANCE_DIR, pt, cfg.TRAIN_SIZE)
        train_paths  = generate_instances(pt, cfg.TRAIN_SIZE,
                                          n_instances=200,   # small for quick test
                                          out_dir=instance_dir)
        val_dir   = os.path.join(cfg.INSTANCE_DIR, pt, "val")
        val_paths = generate_instances(pt, cfg.TRAIN_SIZE,
                                        n_instances=50,
                                        out_dir=val_dir,
                                        seed=cfg.SEED + 1)
    else:
        instance_dir = os.path.join(cfg.INSTANCE_DIR, pt, cfg.TRAIN_SIZE)
        val_dir      = os.path.join(cfg.INSTANCE_DIR, pt, "val")
        train_paths  = [os.path.join(instance_dir, f)
                        for f in os.listdir(instance_dir) if f.endswith('.lp')]
        val_paths    = [os.path.join(val_dir, f)
                        for f in os.listdir(val_dir) if f.endswith('.lp')] \
                       if os.path.exists(val_dir) else []
        print(f"  Found {len(train_paths)} train, {len(val_paths)} val instances")

    # ── Step 2: Collect offline data ───────────────────────────────────────────
    data_path = os.path.join(cfg.DATA_DIR, f"{pt}_data.pkl")
    if not args.skip_collect:
        print(f"\n[2/5] Collecting offline data (hybrid search)...")
        print("  Note: This is slow — K rollouts per node. Reduce K_EXPLORE in config for testing.")
        collect_dataset(train_paths[:50],  # use subset for speed
                        data_path, use_long_term=True)
    else:
        print(f"  Skipping data collection, loading from {data_path}")

    # ── Step 3: Reward assignment ──────────────────────────────────────────────
    print(f"\n[3/5] Assigning rewards...")
    lt_groups, sb_samples = load_dataset(data_path)

    # Long-term reward assignment
    lt_flat = assign_long_term_rewards(lt_groups, top_p=cfg.TOP_P)

    # Short-term reward assignment
    sb_samples = assign_short_term_rewards(sb_samples)

    # Build combined training dataset
    h = {"setcover": 0.70, "auction": 0.90, "facility": 0.95, "indset": 0.90}.get(pt, cfg.SB_PROPORTION)
    combined_samples, train_graphs, train_rewards = build_training_dataset(lt_flat, sb_samples, h=h)

    print(f"  Training samples: {len(train_graphs)} "
          f"(LT: {sum(s.is_long_term for s in combined_samples)}, "
          f"SB: {sum(s.is_short_term for s in combined_samples)})")

    # ── Step 4: Train GCN ──────────────────────────────────────────────────────
    gcn_path = os.path.join(cfg.CHECKPOINT_DIR, f"{pt}_gcn_best.pt")

    if not args.skip_gcn:
        print(f"\n[4/5] Training GCN...")
        gcn = build_gcn(cfg)

        # Initialize prenorm layers from data (Gasse's key trick)
        initialize_prenorms(gcn, train_graphs[:1000])

        trainer = GCNTrainer(gcn, device=device)

        # Simple train/val split from collected data
        split = int(0.9 * len(train_graphs))
        train_data = (combined_samples[:split], train_graphs[:split], train_rewards[:split])
        val_data   = (combined_samples[split:], train_graphs[split:], train_rewards[split:])

        trainer.fit(train_data, val_data, checkpoint_dir=cfg.CHECKPOINT_DIR)
        trainer.save(gcn_path)
    else:
        print(f"  Skipping GCN training")

    # Load GCN for node MLP training
    gcn = build_gcn(cfg)
    if os.path.exists(gcn_path):
        ckpt = torch.load(gcn_path, map_location=device)
        gcn.load_state_dict(ckpt['model_state'])
        gcn.eval()
        print(f"  Loaded GCN from {gcn_path}")

    # ── Step 5: Train NodeMLP ──────────────────────────────────────────────────
    node_mlp_path = os.path.join(cfg.CHECKPOINT_DIR, f"{pt}_node_mlp.pt")

    if not args.skip_node:
        print(f"\n[5/5] Training NodeSelectorMLP...")

        # Build node training labels from trajectory data
        # NOTE: build_node_training_labels expects trajectory dicts with 'node_features'
        # and 'optimal_path'. These need to be collected during a separate solve pass
        # using the trained GCN. This is a TODO for full implementation.
        # For now, we create dummy data to show the pipeline works.
        print("  Building node selection training data...")
        node_data_path = os.path.join(cfg.DATA_DIR, f"{pt}_node_data.pkl")

        if os.path.exists(node_data_path):
            with open(node_data_path, 'rb') as f:
                node_feats, node_labels = pickle.load(f)
            print(f"  Loaded {len(node_feats)} node samples")
        else:
            print("  Node data not found — run evaluate.py with --collect-node-data first")
            print("  Skipping NodeMLP training")
            return

        node_mlp = NodeSelectorMLP()
        trainer  = NodeMLPTrainer(node_mlp, device=device)

        split = int(0.9 * len(node_feats))
        trainer.fit(
            features=node_feats[:split],
            labels=node_labels[:split],
            val_features=node_feats[split:],
            val_labels=node_labels[split:]
        )
        trainer.save(node_mlp_path)
        print(f"  Saved NodeMLP → {node_mlp_path}")
    else:
        print(f"  Skipping NodeMLP training")

    print(f"\nTraining complete. Checkpoints in {cfg.CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
