import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional, List

from runner import train_agent

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(WORKSPACE_DIR, "checkpoints")
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "progress_manifest.json")

SEEDS = [0, 1, 2]

# Configurations adhering strictly to Table 0.5 & Assignment specifications

CONFIG_CARTPOLE = {
    "lr": 5e-4,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 64,
    "learn_every": 1,
    "learning_starts": 64,
    "target_update": "hard",
    "target_update_every": 100,
    "tau": 0.01,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 10_000,
    "loss_type": "mse",
    "hidden_dim": 64,
    "total_episodes": 300,
}

CONFIG_LUNARLANDER = {
    "lr": 5e-4,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 64,
    "learn_every": 4,
    "learning_starts": 64,
    "target_update": "hard",
    "target_update_every": 250,
    "tau": 0.01,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 20_000,
    "loss_type": "mse",
    "hidden_dim": 128,
    "total_episodes": 250,  # 250 episodes ensures completion before 22:35 with 1-hr safety margin
}

CONFIG_PONG = {
    "lr": 1e-4,
    "gamma": 0.99,
    "buffer_size": 100_000,
    "batch_size": 32,
    "learn_every": 1,
    "learning_starts": 10_000,
    "target_update": "hard",
    "target_update_every": 1_000,
    "tau": 0.01,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay_steps": 12_000,
    "loss_type": "huber",
    "grad_clip": 10.0,
    "total_env_steps": 14_000,  # 10k warmup + 4k gradient steps (4 target copies), uniform across all Pong variants
}

CONFIG_TREASUREGRID = {
    "lr": 5e-4,
    "gamma": 0.99,
    "buffer_size": 50_000,
    "batch_size": 64,
    "learn_every": 1,
    "learning_starts": 64,
    "target_update": "hard",
    "target_update_every": 100,
    "tau": 0.01,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 5_000,
    "loss_type": "mse",
    "hidden_dim": 64,
    "total_episodes": 300,
}

# Variants Definition
VARIANTS_EX1_EX2 = {
    # Exercise 1: Ablations
    "DQN_hard": {
        "target_update": "hard",
        "use_replay": True,
        "use_target_network": True,
        "double_dqn": False,
        "dueling": False,
    },
    "DQN_soft": {
        "target_update": "soft",
        "use_replay": True,
        "use_target_network": True,
        "double_dqn": False,
        "dueling": False,
    },
    "DQN_no_replay": {
        "target_update": "hard",
        "use_replay": False,
        "use_target_network": True,
        "double_dqn": False,
        "dueling": False,
    },
    "DQN_no_target": {
        "target_update": "hard",
        "use_replay": True,
        "use_target_network": False,
        "double_dqn": False,
        "dueling": False,
    },
    # Exercise 2: Improvements
    "Double_DQN": {
        "target_update": "hard",
        "use_replay": True,
        "use_target_network": True,
        "double_dqn": True,
        "dueling": False,
    },
    "Dueling_DQN": {
        "target_update": "hard",
        "use_replay": True,
        "use_target_network": True,
        "double_dqn": False,
        "dueling": True,
    },
    "Dueling_Double_DQN": {
        "target_update": "hard",
        "use_replay": True,
        "use_target_network": True,
        "double_dqn": True,
        "dueling": True,
    },
}

VARIANTS_EX3 = {
    "DQN": {
        "target_update": "hard",
        "use_replay": True,
        "use_target_network": True,
        "double_dqn": False,
        "dueling": False,
    },
    "Double_DQN": {
        "target_update": "hard",
        "use_replay": True,
        "use_target_network": True,
        "double_dqn": True,
        "dueling": False,
    },
    "Dueling_DQN": {
        "target_update": "hard",
        "use_replay": True,
        "use_target_network": True,
        "double_dqn": False,
        "dueling": True,
    },
}


def load_manifest() -> Dict[str, Any]:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"experiments": {}, "summary": {"completed": 0, "total": 0}}


def update_manifest(run_id: str, status: str, details: Optional[Dict[str, Any]] = None):
    # Safe manifest update with retry to handle concurrent suite executions
    manifest_file = os.path.join(WORKSPACE_DIR, f"progress_{run_id}.json")
    record = {
        "run_id": run_id,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if details:
        record.update(details)
    with open(manifest_file, "w") as f:
        json.dump(record, f, indent=2)

    # Best-effort aggregated manifest
    for _ in range(5):
        try:
            manifest = load_manifest()
            manifest["experiments"][run_id] = record
            completed = sum(1 for e in manifest["experiments"].values() if e.get("status") == "COMPLETED")
            manifest["summary"] = {
                "completed": completed,
                "total": len(manifest["experiments"]),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2)
            break
        except Exception:
            time.sleep(0.05)


def run_experiment(env_name: str, var_name: str, seed: int, base_cfg: Dict[str, Any], var_cfg: Dict[str, Any]):
    run_id = f"{env_name}_{var_name}_seed{seed}".replace("/", "_")
    result_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{run_id}.pth")

    if os.path.exists(result_path) and os.path.exists(checkpoint_path):
        print(f"[SKIP] {run_id} is already completed.")
        update_manifest(run_id, "COMPLETED", {"result_path": result_path, "checkpoint_path": checkpoint_path})
        return

    update_manifest(run_id, "RUNNING")
    cfg = dict(base_cfg)
    cfg.update(var_cfg)

    t0 = time.time()
    try:
        res = train_agent(env_name, var_name, seed, cfg, save_dir=RESULTS_DIR)
        dt = time.time() - t0
        update_manifest(run_id, "COMPLETED", {
            "elapsed_seconds": dt,
            "train_score": res.get("train_score"),
            "greedy_score": res.get("greedy_score"),
            "result_path": result_path,
            "checkpoint_path": checkpoint_path,
        })
    except Exception as e:
        print(f"[ERROR] Failed {run_id}: {e}")
        update_manifest(run_id, "FAILED", {"error": str(e)})
        raise e


def run_suite(suite_name: str):
    print(f"\n=======================================================")
    print(f"       EXECUTING SUITE: {suite_name.upper()}")
    print(f"=======================================================\n")

    if suite_name in ("cartpole", "all"):
        for var_name, var_cfg in VARIANTS_EX1_EX2.items():
            for s in SEEDS:
                run_experiment("CartPole-v1", var_name, s, CONFIG_CARTPOLE, var_cfg)

    if suite_name in ("treasure", "all"):
        for version in ["v0", "v1", "v2"]:
            env_id = f"TreasureGrid-{version}"
            for var_name, var_cfg in VARIANTS_EX3.items():
                for s in SEEDS:
                    run_experiment(env_id, var_name, s, CONFIG_TREASUREGRID, var_cfg)

    if suite_name in ("lunarlander", "all"):
        for var_name, var_cfg in VARIANTS_EX1_EX2.items():
            for s in SEEDS:
                run_experiment("LunarLander-v3", var_name, s, CONFIG_LUNARLANDER, var_cfg)

    if suite_name in ("pong", "all"):
        for var_name, var_cfg in VARIANTS_EX1_EX2.items():
            for s in SEEDS:
                run_experiment("ALE/Pong-v5", var_name, s, CONFIG_PONG, var_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="cartpole", choices=["cartpole", "treasure", "lunarlander", "pong", "all"])
    args = parser.parse_args()
    run_suite(args.suite)
