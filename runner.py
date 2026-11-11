import os
import json
import time
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

from dqn_core import DQNAgent, build_q_network
import treasure_grid

# Register ALE environments
gym.register_envs(ale_py)


def make_env(env_name: str, seed: Optional[int] = None) -> gym.Env:
    """Factory creating properly wrapped Gymnasium environments."""
    if "Pong" in env_name:
        env = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
        )
        env = FrameStackObservation(env, stack_size=4)
    elif "TreasureGrid" in env_name:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def evaluate_greedy(
    agent: DQNAgent,
    env_name: str,
    n_episodes: int = 20,
    base_seed: int = 100_000,
) -> Tuple[List[float], List[int], float, Dict[str, Any]]:
    """Evaluates greedy policy (epsilon = 0) on n_episodes with fixed evaluation seeds."""
    eval_returns = []
    eval_lengths = []
    successes = []
    bfs_lengths = []
    bfs_returns = []

    is_treasure_grid = "TreasureGrid" in env_name

    for i in range(n_episodes):
        eval_seed = base_seed + i
        env = make_env(env_name, seed=eval_seed)
        state, info = env.reset(seed=eval_seed)

        if is_treasure_grid:
            _, bfs_r, bfs_l = treasure_grid.bfs_optimal_solver(env.unwrapped)
            bfs_lengths.append(bfs_l)
            bfs_returns.append(bfs_r)

        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0
        ep_success = False

        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, step_info = env.step(action)
            ep_ret += reward
            ep_len += 1
            if step_info.get("is_success", False):
                ep_success = True
            state = next_state

        eval_returns.append(ep_ret)
        eval_lengths.append(ep_len)
        successes.append(1.0 if ep_success else 0.0)
        env.close()

    mean_score = float(np.mean(eval_returns))
    extra_metrics = {
        "success_rate": float(np.mean(successes)) if is_treasure_grid else None,
        "mean_length": float(np.mean(eval_lengths)),
        "bfs_mean_length": float(np.mean(bfs_lengths)) if is_treasure_grid else None,
        "bfs_mean_return": float(np.mean(bfs_returns)) if is_treasure_grid else None,
    }
    return eval_returns, eval_lengths, mean_score, extra_metrics


def compute_overestimation_bias(
    agent: DQNAgent,
    env_name: str,
    n_episodes: int = 20,
    base_seed: int = 200_000,
    gamma: float = 0.99,
) -> Dict[str, float]:
    """Computes empirical overestimation bias: mean(Q(s, a) - True_Discounted_Return)."""
    predicted_qs = []
    true_returns = []

    for i in range(n_episodes):
        eval_seed = base_seed + i
        env = make_env(env_name, seed=eval_seed)
        state, _ = env.reset(seed=eval_seed)
        done = False
        truncated = False

        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []

        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            trajectory_states.append(state)
            trajectory_actions.append(action)
            trajectory_rewards.append(reward)
            state = next_state
        env.close()

        # Compute discounted return G_t for each step in trajectory
        T = len(trajectory_rewards)
        if T == 0:
            continue
        G = np.zeros(T)
        running_add = 0.0
        for t in reversed(range(T)):
            running_add = trajectory_rewards[t] + gamma * running_add
            G[t] = running_add

        # Predict Q(s_t, a_t) under agent network
        # Sample up to 10 points per episode to keep evaluation balanced
        sample_indices = np.linspace(0, T - 1, min(T, 10), dtype=int)
        for idx in sample_indices:
            s = trajectory_states[idx]
            a = trajectory_actions[idx]
            with torch.no_grad():
                st = torch.as_tensor(s, dtype=torch.uint8 if len(agent.state_shape) == 3 else torch.float32, device=agent.device)
                if st.ndim == len(agent.state_shape):
                    st = st.unsqueeze(0)
                q_vals = agent.q_net(st)
                q_pred = q_vals[0, a].item()

            predicted_qs.append(q_pred)
            true_returns.append(G[idx])

    if len(predicted_qs) == 0:
        return {"mean_bias": 0.0, "mean_q": 0.0, "mean_true": 0.0}

    predicted_qs = np.array(predicted_qs)
    true_returns = np.array(true_returns)
    biases = predicted_qs - true_returns

    return {
        "mean_bias": float(np.mean(biases)),
        "mean_q": float(np.mean(predicted_qs)),
        "mean_true": float(np.mean(true_returns)),
        "std_bias": float(np.std(biases)),
    }


def train_agent(
    env_name: str,
    variant_name: str,
    seed: int,
    config: Dict[str, Any],
    save_dir: str = "C:/Users/achou/Documents/DQN_Homework_2026/results",
) -> Dict[str, Any]:
    """Trains agent on environment according to configuration and logs all metrics.
    
    Robust against interruptions:
    1. If final .json and .pth exist, skips instantly.
    2. Periodically saves intermediate checkpoint to allow resuming mid-training.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoints_dir = os.path.join(os.path.dirname(save_dir), "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    run_id = f"{env_name}_{variant_name}_seed{seed}".replace("/", "_")
    result_path = os.path.join(save_dir, f"{run_id}.json")
    checkpoint_path = os.path.join(checkpoints_dir, f"{run_id}.pth")
    temp_ckpt_path = os.path.join(checkpoints_dir, f"{run_id}_inprogress.pt")

    # 1. Skip if already completed
    if os.path.exists(result_path) and os.path.exists(checkpoint_path):
        print(f"[CACHE] Found completed results for {run_id}, loading...")
        with open(result_path, "r") as f:
            return json.load(f)

    print(f"\n[START] Training {run_id} ...")
    start_time = time.time()

    # Create environment
    env = make_env(env_name, seed=seed)
    state, _ = env.reset(seed=seed)

    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # Build agent
    agent_kwargs = {
        "state_shape": state_shape,
        "action_dim": action_dim,
        "env_name": env_name,
        "lr": config.get("lr", 5e-4),
        "gamma": config.get("gamma", 0.99),
        "buffer_size": config.get("buffer_size", 100_000),
        "batch_size": config.get("batch_size", 64),
        "learn_every": config.get("learn_every", 1),
        "learning_starts": config.get("learning_starts", 64),
        "target_update": config.get("target_update", "hard"),
        "target_update_every": config.get("target_update_every", 100),
        "tau": config.get("tau", 0.01),
        "eps_start": config.get("eps_start", 1.0),
        "eps_end": config.get("eps_end", 0.05),
        "eps_decay_steps": config.get("eps_decay_steps", 10_000),
        "loss_type": config.get("loss_type", "mse"),
        "grad_clip": config.get("grad_clip", None),
        "use_replay": config.get("use_replay", True),
        "use_target_network": config.get("use_target_network", True),
        "double_dqn": config.get("double_dqn", False),
        "dueling": config.get("dueling", False),
        "hidden_dim": config.get("hidden_dim", 64),
        "seed": seed,
    }

    agent = DQNAgent(**agent_kwargs)

    is_step_budget = "total_env_steps" in config
    total_budget_steps = config.get("total_env_steps", 1_000_000)
    total_budget_episodes = config.get("total_episodes", 300)

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    step_records: List[int] = []
    ep_count = 0

    # 2. Check for in-progress checkpoint to resume
    if os.path.exists(temp_ckpt_path):
        try:
            print(f"[RESUME] Found in-progress checkpoint for {run_id}, resuming...")
            saved = torch.load(temp_ckpt_path, map_location=agent.device, weights_only=False)
            agent.q_net.load_state_dict(saved["q_net"])
            agent.target_net.load_state_dict(saved["target_net"])
            agent.optimizer.load_state_dict(saved["optimizer"])
            agent.total_env_steps = saved["total_env_steps"]
            agent.total_gradient_steps = saved["total_gradient_steps"]
            agent.losses = saved["losses"]
            agent.q_means = saved["q_means"]
            ep_count = saved["ep_count"]
            episode_returns = saved["episode_returns"]
            episode_lengths = saved["episode_lengths"]
            step_records = saved["step_records"]
            print(f"[RESUME] Restored at episode {ep_count}, step {agent.total_env_steps}.")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint {temp_ckpt_path}: {e}. Restarting run.")

    ep_ret = 0.0
    ep_len = 0

    checkpoint_interval_eps = 50
    checkpoint_interval_steps = 2500

    last_saved_step = agent.total_env_steps

    while True:
        # Check termination condition
        if is_step_budget:
            if agent.total_env_steps >= total_budget_steps:
                break
        else:
            if ep_count >= total_budget_episodes:
                break

        action = agent.select_action(state, evaluate=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # For Pong, clip reward to sign(r) during learning while reporting true score
        learning_reward = np.sign(reward) if "Pong" in env_name else reward

        agent.step(state, action, learning_reward, next_state, terminated)

        state = next_state
        ep_ret += reward
        ep_len += 1

        if done:
            ep_count += 1
            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            step_records.append(agent.total_env_steps)

            if ep_count % 20 == 0 or (is_step_budget and agent.total_env_steps % 5000 == 0):
                avg100 = np.mean(episode_returns[-min(len(episode_returns), 100):])
                print(
                    f"[{run_id}] Ep {ep_count} | Step {agent.total_env_steps} | "
                    f"Last Ep Ret: {ep_ret:.1f} | Avg Ret: {avg100:.1f} | eps: {agent.epsilon:.3f}"
                )

            # Periodic checkpointing
            if ep_count % checkpoint_interval_eps == 0:
                torch.save({
                    "q_net": agent.q_net.state_dict(),
                    "target_net": agent.target_net.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "total_env_steps": agent.total_env_steps,
                    "total_gradient_steps": agent.total_gradient_steps,
                    "losses": agent.losses,
                    "q_means": agent.q_means,
                    "ep_count": ep_count,
                    "episode_returns": episode_returns,
                    "episode_lengths": episode_lengths,
                    "step_records": step_records,
                }, temp_ckpt_path)

            state, _ = env.reset()
            ep_ret = 0.0
            ep_len = 0

        # Step-based periodic checkpointing for long/step-based runs (Pong)
        if is_step_budget and (agent.total_env_steps - last_saved_step) >= checkpoint_interval_steps:
            last_saved_step = agent.total_env_steps
            torch.save({
                "q_net": agent.q_net.state_dict(),
                "target_net": agent.target_net.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "total_env_steps": agent.total_env_steps,
                "total_gradient_steps": agent.total_gradient_steps,
                "losses": agent.losses,
                "q_means": agent.q_means,
                "ep_count": ep_count,
                "episode_returns": episode_returns,
                "episode_lengths": episode_lengths,
                "step_records": step_records,
            }, temp_ckpt_path)

    env.close()

    elapsed = time.time() - start_time
    print(f"[DONE] {run_id} finished in {elapsed:.1f}s ({elapsed/60:.2f} min).")

    # Greedy evaluation on 20 unseen seeds (100_000 + i)
    eval_returns, eval_lengths, greedy_score, extra_metrics = evaluate_greedy(
        agent,
        env_name=env_name,
        n_episodes=20,
        base_seed=100_000,
    )

    # Overestimation bias measurement
    bias_metrics = compute_overestimation_bias(
        agent,
        env_name=env_name,
        n_episodes=20,
        base_seed=200_000,
        gamma=config.get("gamma", 0.99),
    )

    # Calculate training score convention
    if "Pong" in env_name:
        train_score = float(np.mean(episode_returns[-min(len(episode_returns), 20):]))
    else:
        train_score = float(np.mean(episode_returns[-min(len(episode_returns), 100):]))

    # Downsample q_means and losses for storage if very long
    max_pts = 2000
    if len(agent.q_means) > max_pts:
        step_stride = len(agent.q_means) // max_pts
        stored_q_means = agent.q_means[::step_stride]
        stored_losses = agent.losses[::step_stride]
    else:
        stored_q_means = agent.q_means
        stored_losses = agent.losses

    # Save permanent model weights
    agent.save(checkpoint_path)

    # Clean up temporary in-progress file
    if os.path.exists(temp_ckpt_path):
        try:
            os.remove(temp_ckpt_path)
        except OSError:
            pass

    results = {
        "run_id": run_id,
        "env_name": env_name,
        "variant_name": variant_name,
        "seed": seed,
        "config": config,
        "elapsed_seconds": elapsed,
        "total_env_steps": agent.total_env_steps,
        "total_gradient_steps": agent.total_gradient_steps,
        "total_episodes": ep_count,
        "train_score": train_score,
        "greedy_score": greedy_score,
        "extra_metrics": extra_metrics,
        "eval_returns": eval_returns,
        "eval_lengths": eval_lengths,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "step_records": step_records,
        "q_means": stored_q_means,
        "losses": stored_losses,
        "bias_metrics": bias_metrics,
        "checkpoint_path": checkpoint_path,
    }

    with open(result_path, "w") as f:
        json.dump(results, f)

    return results

    with open(result_path, "w") as f:
        json.dump(results, f)

    return results
