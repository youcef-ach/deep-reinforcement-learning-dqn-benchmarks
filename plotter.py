import os
import glob
import json
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_runs(env_name: str, variant_name: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(RESULTS_DIR, f"{env_name.replace('/', '_')}_{variant_name}_seed*.json")
    files = glob.glob(pattern)
    runs = []
    for f in sorted(files):
        with open(f, "r") as fp:
            runs.append(json.load(fp))
    return runs


def smooth_curve(arr: np.ndarray, window: int = 20) -> np.ndarray:
    if len(arr) < window:
        return arr
    smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
    # Pad beginning to keep same length
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def plot_shaded_curve(
    ax: plt.Axes,
    x: np.ndarray,
    curves: List[np.ndarray],
    label: str,
    color: str,
    linestyle: str = "-",
):
    # Align curves to minimum common length
    min_len = min(len(c) for c in curves)
    aligned = np.array([c[:min_len] for c in curves])
    x_aligned = x[:min_len]

    mean_c = np.mean(aligned, axis=0)
    min_c = np.min(aligned, axis=0)
    max_c = np.max(aligned, axis=0)

    ax.plot(x_aligned, mean_c, label=label, color=color, linestyle=linestyle, linewidth=2.0)
    ax.fill_between(x_aligned, min_c, max_c, color=color, alpha=0.18)


def generate_ex1_plots(env_name: str, save_prefix: str, solved_score: Optional[float] = None, q_ref: Optional[float] = None):
    variants = ["DQN_hard", "DQN_soft", "DQN_no_replay", "DQN_no_target"]
    colors = {
        "DQN_hard": "#1f77b4",       # Blue
        "DQN_soft": "#2ca02c",       # Green
        "DQN_no_replay": "#d62728",   # Red
        "DQN_no_target": "#ff7f0e",   # Orange
    }
    labels = {
        "DQN_hard": "DQN (Hard Update, Ref)",
        "DQN_soft": r"DQN (Soft Update, $\tau=0.01$)",
        "DQN_no_replay": "No Experience Replay",
        "DQN_no_target": "No Target Network",
    }

    # 1. Learning curves
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for var in variants:
        runs = load_runs(env_name, var)
        if not runs:
            continue
        all_smoothed = [smooth_curve(np.array(r["episode_returns"]), window=20) for r in runs]
        max_eps = max(len(s) for s in all_smoothed)
        x = np.arange(1, max_eps + 1)
        plot_shaded_curve(ax, x, all_smoothed, label=labels[var], color=colors[var])

    if solved_score is not None:
        ax.axhline(solved_score, color="black", linestyle="--", alpha=0.7, label=f"Solved Threshold ({solved_score})")

    ax.set_title(f"Exercise 1: DQN Ablations on {env_name} (Seeds 0, 1, 2)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (20-ep moving avg)")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_learning_curves.png"), dpi=300)
    plt.close(fig)

    # 2. Predicted Q-values during training
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for var in variants:
        runs = load_runs(env_name, var)
        if not runs:
            continue
        q_curves = [np.array(r["q_means"]) for r in runs if len(r.get("q_means", [])) > 0]
        if not q_curves:
            continue
        max_len = max(len(q) for q in q_curves)
        x = np.linspace(0, 100, max_len)  # normalized % of training
        plot_shaded_curve(ax, x, q_curves, label=labels[var], color=colors[var])

    if q_ref is not None:
        ax.axhline(q_ref, color="red", linestyle=":", linewidth=2, label=rf"Theoretical Limit $1/(1-\gamma) = {q_ref:.0f}$")

    ax.set_title(f"Exercise 1: Mean Predicted Q-Values on {env_name}")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel(r"Mean Predicted $Q(s, a; \theta)$")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_q_values.png"), dpi=300)
    plt.close(fig)


def generate_ex2_plots(env_name: str, save_prefix: str, solved_score: Optional[float] = None):
    variants = ["DQN_hard", "Double_DQN", "Dueling_DQN", "Dueling_Double_DQN"]
    colors = {
        "DQN_hard": "#1f77b4",            # Blue
        "Double_DQN": "#2ca02c",          # Green
        "Dueling_DQN": "#9467bd",         # Purple
        "Dueling_Double_DQN": "#e377c2",  # Pink
    }
    labels = {
        "DQN_hard": "DQN (Baseline)",
        "Double_DQN": "Double DQN",
        "Dueling_DQN": "Dueling DQN",
        "Dueling_Double_DQN": "Dueling Double DQN",
    }

    # 1. Learning curves
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for var in variants:
        runs = load_runs(env_name, var)
        if not runs:
            continue
        all_smoothed = [smooth_curve(np.array(r["episode_returns"]), window=20) for r in runs]
        max_eps = max(len(s) for s in all_smoothed)
        x = np.arange(1, max_eps + 1)
        plot_shaded_curve(ax, x, all_smoothed, label=labels[var], color=colors[var])

    if solved_score is not None:
        ax.axhline(solved_score, color="black", linestyle="--", alpha=0.7, label=f"Solved Threshold ({solved_score})")

    ax.set_title(f"Exercise 2: Advanced DQN Architectures on {env_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (20-ep moving avg)")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_comparison_curves.png"), dpi=300)
    plt.close(fig)

    # 2. Predicted Q-values
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for var in variants:
        runs = load_runs(env_name, var)
        if not runs:
            continue
        q_curves = [np.array(r["q_means"]) for r in runs if len(r.get("q_means", [])) > 0]
        if not q_curves:
            continue
        max_len = max(len(q) for q in q_curves)
        x = np.linspace(0, 100, max_len)
        plot_shaded_curve(ax, x, q_curves, label=labels[var], color=colors[var])

    ax.set_title(f"Exercise 2: Predicted Q-Values on {env_name}")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel(r"Mean Predicted $Q(s, a; \theta)$")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{save_prefix}_q_values.png"), dpi=300)
    plt.close(fig)


def generate_overestimation_bias_plot():
    envs = ["CartPole-v1", "LunarLander-v3", "ALE/Pong-v5"]
    variants = ["DQN_hard", "Double_DQN", "Dueling_DQN", "Dueling_Double_DQN"]
    short_labels = ["DQN", "Double DQN", "Dueling DQN", "Dueling DDQN"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(envs))
    width = 0.2

    for idx, var in enumerate(variants):
        means = []
        errs = []
        for env in envs:
            runs = load_runs(env, var)
            if runs:
                biases = [r["bias_metrics"]["mean_bias"] for r in runs if "bias_metrics" in r]
                means.append(np.mean(biases) if biases else 0.0)
                errs.append((np.max(biases) - np.min(biases)) / 2.0 if len(biases) > 1 else 0.0)
            else:
                means.append(0.0)
                errs.append(0.0)

        offset = (idx - 1.5) * width
        ax.bar(x + offset, means, width, yerr=errs, capsize=4, label=short_labels[idx])

    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["CartPole-v1", "LunarLander-v3", "ALE/Pong-v5"])
    ax.set_ylabel(r"Empirical Overestimation Bias: $\mathbb{E}[Q(s, a) - G_t]$")
    ax.set_title("Overestimation Bias Across Environments (Mean & Range across 3 Seeds)")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ex2_overestimation_bias.png"), dpi=300)
    plt.close(fig)


def generate_ex3_plots():
    versions = ["TreasureGrid-v0", "TreasureGrid-v1", "TreasureGrid-v2"]
    v_labels = ["v0 (Sparse, Coords)", "v1 (Shaped, Norm Coords)", "v2 (Sparse, Spatial Grid)"]
    agents = ["DQN", "Double_DQN", "Dueling_DQN"]
    colors = {"DQN": "#1f77b4", "Double_DQN": "#2ca02c", "Dueling_DQN": "#9467bd"}

    # 1. Learning curves per version (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for i, v in enumerate(versions):
        ax = axes[i]
        for ag in agents:
            runs = load_runs(v, ag)
            if not runs:
                continue
            all_smoothed = [smooth_curve(np.array(r["episode_returns"]), window=20) for r in runs]
            x = np.arange(1, max(len(s) for s in all_smoothed) + 1)
            plot_shaded_curve(ax, x, all_smoothed, label=ag, color=colors[ag])
        ax.set_title(v_labels[i])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (20-ep moving avg)")
        ax.legend(loc="best", framealpha=0.9)

    plt.suptitle("Exercise 3: TreasureGrid Learning Curves Across Environment Versions", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ex3_customenv_learning_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Benchmark vs BFS and Random
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ["Greedy Score", "Success Rate (%)", "Episode Length"]

    for i, v in enumerate(versions):
        # Gather data
        ag_scores = []
        ag_success = []
        ag_len = []
        labels = []

        # Baselines
        bfs_len = None
        bfs_ret = None

        for ag in agents:
            runs = load_runs(v, ag)
            if runs:
                ag_scores.append(np.mean([r["greedy_score"] for r in runs]))
                succ = [r["extra_metrics"].get("success_rate", 0.0) * 100 for r in runs if "extra_metrics" in r]
                ag_success.append(np.mean(succ) if succ else 0.0)
                l = [r["extra_metrics"].get("mean_length", 0.0) for r in runs if "extra_metrics" in r]
                ag_len.append(np.mean(l) if l else 0.0)
                labels.append(ag)
                if bfs_len is None and "extra_metrics" in runs[0]:
                    bfs_len = runs[0]["extra_metrics"].get("bfs_mean_length", 7.0)
                    bfs_ret = runs[0]["extra_metrics"].get("bfs_mean_return", 9.9)

        # Plot Subplot i: Success rates & Episode Lengths
        ax = axes[i]
        x = np.arange(len(labels))
        bars = ax.bar(x, ag_success, width=0.45, color=["#1f77b4", "#2ca02c", "#9467bd"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{v_labels[i]}\n(BFS Opt Length: {bfs_len:.1f} steps)")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    plt.suptitle("Exercise 3: Navigation Success Rate vs Baselines (20 Greedy Evaluation Episodes)", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ex3_customenv_benchmarks.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots():
    print("Generating all publication plots...")
    generate_ex1_plots("CartPole-v1", "ex1_cartpole", solved_score=475, q_ref=100.0)
    generate_ex1_plots("LunarLander-v3", "ex1_lunarlander", solved_score=200)
    generate_ex1_plots("ALE/Pong-v5", "ex1_pong", solved_score=18)

    generate_ex2_plots("CartPole-v1", "ex2_cartpole", solved_score=475)
    generate_ex2_plots("LunarLander-v3", "ex2_lunarlander", solved_score=200)
    generate_ex2_plots("ALE/Pong-v5", "ex2_pong", solved_score=18)

    generate_overestimation_bias_plot()
    generate_ex3_plots()
    print("All plots generated successfully in", FIGURES_DIR)


if __name__ == "__main__":
    generate_all_plots()

