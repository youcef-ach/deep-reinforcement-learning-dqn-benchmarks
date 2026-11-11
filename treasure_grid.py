from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TreasureGridEnv(gym.Env):
    """Custom TreasureGrid Gymnasium environment.
    
    Grid of size N x N with walls, 2 traps, and 1 treasure.
    Supports 3 distinct versions:
    - version='v0': Coordinate observation (8,), sparse reward (+10 treasure, -10 trap, -0.01 step)
    - version='v1': Normalized coordinate observation (8,), distance-shaped reward
    - version='v2': Flat spatial grid channels (N*N*4,), sparse reward (+15 treasure, -15 trap, -0.05 step)
    
    Actions:
      0: Up (row - 1)
      1: Down (row + 1)
      2: Left (col - 1)
      3: Right (col + 1)
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    # Action displacements: (d_row, d_col)
    ACTION_DELTAS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }

    def __init__(
        self,
        n: int = 6,
        version: str = "v0",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.n = n
        self.version = version
        self.render_mode = render_mode
        self.max_steps = 4 * (n ** 2)  # Truncated after 4 N^2 steps

        self.action_space = spaces.Discrete(4)

        if version in ("v0", "v1"):
            # State: (agent_r, agent_c, treasure_r, treasure_c, trap1_r, trap1_c, trap2_r, trap2_c)
            low = 0.0
            high = 1.0 if version == "v1" else float(n - 1)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                shape=(8,),
                dtype=np.float32,
            )
        elif version == "v2":
            # 4 channels: agent, treasure, traps, walls -> flattened N * N * 4
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n * n * 4,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown version: {version}")

        # Static walls layout (interior obstacles)
        self.walls: Set[Tuple[int, int]] = {
            (1, 2),
            (2, 2),
            (3, 2),
            (3, 4),
            (4, 4),
        }

        # Dynamic state entities
        self.agent_pos = (0, 0)
        self.treasure_pos = (n - 1, n - 1)
        self.trap_positions: List[Tuple[int, int]] = []
        self.step_count = 0
        self.prev_dist = 0.0

    def _get_obs(self) -> np.ndarray:
        if self.version == "v0":
            obs = np.array(
                [
                    self.agent_pos[0],
                    self.agent_pos[1],
                    self.treasure_pos[0],
                    self.treasure_pos[1],
                    self.trap_positions[0][0],
                    self.trap_positions[0][1],
                    self.trap_positions[1][0],
                    self.trap_positions[1][1],
                ],
                dtype=np.float32,
            )
            return obs

        elif self.version == "v1":
            norm = float(self.n - 1)
            obs = np.array(
                [
                    self.agent_pos[0] / norm,
                    self.agent_pos[1] / norm,
                    self.treasure_pos[0] / norm,
                    self.treasure_pos[1] / norm,
                    self.trap_positions[0][0] / norm,
                    self.trap_positions[0][1] / norm,
                    self.trap_positions[1][0] / norm,
                    self.trap_positions[1][1] / norm,
                ],
                dtype=np.float32,
            )
            return obs

        elif self.version == "v2":
            grid = np.zeros((4, self.n, self.n), dtype=np.float32)
            # Channel 0: Agent
            grid[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
            # Channel 1: Treasure
            grid[1, self.treasure_pos[0], self.treasure_pos[1]] = 1.0
            # Channel 2: Traps
            for tr, tc in self.trap_positions:
                grid[2, tr, tc] = 1.0
            # Channel 3: Walls
            for wr, wc in self.walls:
                grid[3, wr, wc] = 1.0
            return grid.flatten()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0

        # Available open cells
        all_cells = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in self.walls]
        self.np_random.shuffle(all_cells)

        if self.version == "v0":
            # Fixed treasure position
            self.treasure_pos = (self.n - 1, self.n - 1)
            candidates = [c for c in all_cells if c != self.treasure_pos]
            self.agent_pos = candidates[0]
            self.trap_positions = [candidates[1], candidates[2]]
        elif self.version == "v1":
            # Dynamic treasure position
            self.treasure_pos = all_cells[0]
            self.agent_pos = all_cells[1]
            self.trap_positions = [all_cells[2], all_cells[3]]
        else:  # v2
            # Dynamic positions
            self.treasure_pos = all_cells[0]
            self.agent_pos = all_cells[1]
            self.trap_positions = [all_cells[2], all_cells[3]]

        self.prev_dist = abs(self.agent_pos[0] - self.treasure_pos[0]) + abs(self.agent_pos[1] - self.treasure_pos[1])

        obs = self._get_obs()
        info = {
            "agent_pos": self.agent_pos,
            "treasure_pos": self.treasure_pos,
            "trap_positions": self.trap_positions,
            "is_success": False,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        d_r, d_c = self.ACTION_DELTAS[action]
        next_r = self.agent_pos[0] + d_r
        next_c = self.agent_pos[1] + d_c

        # Wall or boundary collision -> stay in place
        if 0 <= next_r < self.n and 0 <= next_c < self.n and (next_r, next_c) not in self.walls:
            self.agent_pos = (next_r, next_c)

        curr_dist = abs(self.agent_pos[0] - self.treasure_pos[0]) + abs(self.agent_pos[1] - self.treasure_pos[1])

        terminated = False
        is_success = False
        reward = 0.0

        if self.agent_pos == self.treasure_pos:
            terminated = True
            is_success = True
            if self.version == "v2":
                reward = 15.0
            else:
                reward = 10.0
        elif self.agent_pos in self.trap_positions:
            terminated = True
            is_success = False
            if self.version == "v2":
                reward = -15.0
            else:
                reward = -10.0
        else:
            if self.version == "v0":
                reward = -0.01  # small step penalty
            elif self.version == "v1":
                # Shaped potential: progress towards treasure
                progress = float(self.prev_dist - curr_dist)
                reward = progress * 0.5 - 0.01
            elif self.version == "v2":
                reward = -0.05

        self.prev_dist = curr_dist
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "agent_pos": self.agent_pos,
            "treasure_pos": self.treasure_pos,
            "trap_positions": self.trap_positions,
            "is_success": is_success,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            grid = [["." for _ in range(self.n)] for _ in range(self.n)]
            for wr, wc in self.walls:
                grid[wr][wc] = "#"
            for tr, tc in self.trap_positions:
                grid[tr][tc] = "X"
            tr, tc = self.treasure_pos
            grid[tr][tc] = "T"
            ar, ac = self.agent_pos
            grid[ar][ac] = "A"
            return "\n".join(" ".join(row) for row in grid)
        elif self.render_mode == "rgb_array":
            # Return RGB array representation of the grid (scale factor e.g. 16 pixels per cell)
            scale = 16
            img = np.ones((self.n * scale, self.n * scale, 3), dtype=np.uint8) * 240  # light grey background
            def draw_rect(r, c, color):
                img[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = color
            for wr, wc in self.walls:
                draw_rect(wr, wc, [80, 80, 80])  # dark grey walls
            for tr, tc in self.trap_positions:
                draw_rect(tr, tc, [220, 50, 50])  # red traps
            tr, tc = self.treasure_pos
            draw_rect(tr, tc, [255, 215, 0])  # gold treasure
            ar, ac = self.agent_pos
            draw_rect(ar, ac, [50, 100, 240])  # blue agent
            return img


def bfs_optimal_solver(env: TreasureGridEnv) -> Tuple[Optional[List[int]], float, int]:
    """Computes optimal BFS shortest path avoiding walls and traps.
    
    Returns:
      (action_sequence, total_reward, path_length)
      If unreachable, returns (None, -truncated_penalty, max_steps)
    """
    start = env.agent_pos
    goal = env.treasure_pos
    traps = set(env.trap_positions)
    walls = set(env.walls)

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        curr, path = queue.popleft()
        if curr == goal:
            # Reconstruct actions and reward
            actions = path
            length = len(actions)
            if env.version == "v0":
                tot_reward = 10.0 - 0.01 * (length - 1)
            elif env.version == "v1":
                # Distance reduction + goal
                tot_reward = 10.0 - 0.01 * (length - 1) + 0.5 * (
                    abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                )
            elif env.version == "v2":
                tot_reward = 15.0 - 0.05 * (length - 1)
            return actions, tot_reward, length

        for a, (dr, dc) in env.ACTION_DELTAS.items():
            nr, nc = curr[0] + dr, curr[1] + dc
            next_cell = (nr, nc)
            if (
                0 <= nr < env.n
                and 0 <= nc < env.n
                and next_cell not in walls
                and next_cell not in traps
                and next_cell not in visited
            ):
                visited.add(next_cell)
                queue.append((next_cell, path + [a]))

    # Unreachable without hitting a trap
    return None, -10.0, env.max_steps


# Register environments with Gymnasium
for v in ["v0", "v1", "v2"]:
    env_id = f"TreasureGrid-{v}"
    if env_id not in gym.envs.registration.registry:
        gym.register(
            id=env_id,
            entry_point="treasure_grid:TreasureGridEnv",
            kwargs={"version": v},
        )
