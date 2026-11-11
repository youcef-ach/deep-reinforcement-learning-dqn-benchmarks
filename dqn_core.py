import random
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    """Experience replay buffer.
    
    Supports both standard uniform random sampling and the 'no experience replay'
    mode (keeping only the last batch_size transitions and returning consecutive transitions).
    Supports uint8 frame observations for memory efficiency (e.g. Atari Pong).
    """

    def __init__(
        self,
        capacity: int,
        state_dim: Union[int, Tuple[int, ...]],
        batch_size: int = 64,
        dtype: np.dtype = np.float32,
        use_replay: bool = True,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.use_replay = use_replay
        self.dtype = dtype

        if isinstance(state_dim, int):
            self.state_shape = (state_dim,)
        else:
            self.state_shape = tuple(state_dim)

        if not self.use_replay:
            # Memory of size B: keeps only the last batch_size transitions
            self.capacity = batch_size

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=self.dtype)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=self.dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.states[self.pos] = np.asarray(state, dtype=self.dtype)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = np.asarray(next_state, dtype=self.dtype)
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_replay:
            # Minibatch is the B most recent consecutive transitions
            assert self.size == self.capacity, f"Buffer has only {self.size}/{self.capacity} transitions"
            # Order from oldest to newest in circular buffer
            indices = [(self.pos + i) % self.capacity for i in range(self.capacity)]
        else:
            b = batch_size if batch_size is not None else self.batch_size
            assert self.size >= b, f"Cannot sample {b} from buffer of size {self.size}"
            indices = np.random.choice(self.size, size=b, replace=False)

        states = torch.from_numpy(self.states[indices])
        actions = torch.from_numpy(self.actions[indices])
        rewards = torch.from_numpy(self.rewards[indices])
        next_states = torch.from_numpy(self.next_states[indices])
        dones = torch.from_numpy(self.dones[indices])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


class MLPQNetwork(nn.Module):
    """Standard MLP Q-network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        return self.net(x)


class NatureCNNQNetwork(nn.Module):
    """Nature CNN Q-network (Mnih et al., 2015).
    
    conv 32 x 8x8 stride 4 -> conv 64 x 4x4 stride 2 -> conv 64 x 3x3 stride 1 -> FC 512 (3136 -> 512), ReLU
    Linear -> 6 Q-values (exactly 1,687,206 parameters for 6 actions).
    Input uint8 frames [0, 255] are divided by 255.0 inside network.
    """

    def __init__(self, in_channels: int = 4, action_dim: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        features = self.conv(x)
        return self.fc(features)


class MLPDuelingQNetwork(nn.Module):
    """Dueling MLP Q-network (Wang et al., 2016).
    
    Shared body: Linear -> ReLU
    Value stream V: Linear -> ReLU -> Linear(..., 1)
    Advantage stream A: Linear -> ReLU -> Linear(..., action_dim)
    Q(s, a) = V(s) + A(s, a) - (1 / |A|) * sum_{a'} A(s, a')
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        feat = self.shared(x)
        val = self.value_stream(feat)  # (B, 1)
        adv = self.advantage_stream(feat)  # (B, A)
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q


class NatureCNNDuelingQNetwork(nn.Module):
    """Dueling Nature CNN Q-network.
    
    Shared body: 3 Nature CNN convolutions (-> 3136)
    Value stream V: 3136 -> 512 -> 1
    Advantage stream A: 3136 -> 512 -> action_dim
    """

    def __init__(self, in_channels: int = 4, action_dim: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        feat = self.conv(x)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q


def build_q_network(
    env_name: str,
    state_shape: Union[int, Tuple[int, ...]],
    action_dim: int,
    dueling: bool = False,
    hidden_dim: Optional[int] = None,
) -> nn.Module:
    is_image = len(state_shape) == 3 if isinstance(state_shape, tuple) else False

    if is_image:
        in_channels = state_shape[0]
        if dueling:
            return NatureCNNDuelingQNetwork(in_channels=in_channels, action_dim=action_dim)
        else:
            return NatureCNNQNetwork(in_channels=in_channels, action_dim=action_dim)
    else:
        state_dim = state_shape[0] if isinstance(state_shape, tuple) else state_shape
        h_dim = hidden_dim if hidden_dim is not None else 64
        if dueling:
            return MLPDuelingQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=h_dim)
        else:
            return MLPQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=h_dim)


class DQNAgent:
    """Unified DQN Agent supporting all homework variants.
    
    Ablations & Enhancements:
    - target_update: 'hard' (copy weights every C gradient steps) or 'soft' (tau)
    - use_replay: bool (if False, uses consecutive minibatch memory of size batch_size)
    - use_target_network: bool (if False, online network computes targets)
    - double_dqn: bool (if True, Double Q-learning target)
    - dueling: bool (if True, Dueling network architecture)
    """

    def __init__(
        self,
        state_shape: Union[int, Tuple[int, ...]],
        action_dim: int,
        env_name: str = "CartPole-v1",
        lr: float = 5e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        learn_every: int = 1,
        learning_starts: int = 64,
        target_update: str = "hard",  # 'hard' or 'soft'
        target_update_every: int = 100,  # C gradient steps
        tau: float = 0.01,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10_000,
        loss_type: str = "mse",  # 'mse' or 'huber'
        grad_clip: Optional[float] = None,
        use_replay: bool = True,
        use_target_network: bool = True,
        double_dqn: bool = False,
        dueling: bool = False,
        hidden_dim: Optional[int] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.state_shape = (state_shape,) if isinstance(state_shape, int) else tuple(state_shape)
        self.action_dim = action_dim
        self.env_name = env_name
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.learning_starts = learning_starts
        self.target_update = target_update
        self.target_update_every = target_update_every
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.loss_type = loss_type
        self.grad_clip = grad_clip
        self.use_replay = use_replay
        self.use_target_network = use_target_network
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.device = torch.device(device)

        # Networks
        self.q_net = build_q_network(
            env_name=env_name,
            state_shape=self.state_shape,
            action_dim=action_dim,
            dueling=dueling,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_net = build_q_network(
            env_name=env_name,
            state_shape=self.state_shape,
            action_dim=action_dim,
            dueling=dueling,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        dtype = np.uint8 if len(self.state_shape) == 3 else np.float32
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=self.state_shape,
            batch_size=batch_size,
            dtype=dtype,
            use_replay=use_replay,
        )

        # Tracking state
        self.total_env_steps = 0
        self.total_gradient_steps = 0
        self.losses: List[float] = []
        self.q_means: List[float] = []

    @property
    def epsilon(self) -> float:
        if self.eps_decay_steps <= 0:
            return self.eps_end
        fraction = min(1.0, self.total_env_steps / float(self.eps_decay_steps))
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.uint8 if len(self.state_shape) == 3 else torch.float32, device=self.device)
            if state_t.ndim == len(self.state_shape):
                state_t = state_t.unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=-1).item())

    def compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target Q-values y_i according to variant."""
        with torch.no_grad():
            if not self.use_target_network:
                # No target network: use online network q_net
                next_q = self.q_net(next_states)
                max_next_q, _ = next_q.max(dim=-1)
                targets = rewards + self.gamma * (1.0 - dones) * max_next_q
            elif self.double_dqn:
                # Double DQN: online network selects action, target network evaluates
                next_q_online = self.q_net(next_states)
                best_actions = next_q_online.argmax(dim=-1, keepdim=True)
                next_q_target = self.target_net(next_states)
                double_q = next_q_target.gather(dim=-1, index=best_actions).squeeze(-1)
                targets = rewards + self.gamma * (1.0 - dones) * double_q
            else:
                # Standard DQN with target network
                next_q_target = self.target_net(next_states)
                max_next_q, _ = next_q_target.max(dim=-1)
                targets = rewards + self.gamma * (1.0 - dones) * max_next_q
        return targets

    def update(self) -> Optional[Tuple[float, float]]:
        if len(self.buffer) < self.learning_starts or len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Online predicted Q-values
        q_pred_all = self.q_net(states)
        q_pred = q_pred_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        # Target Q-values
        targets = self.compute_targets(rewards, next_states, dones)

        # Loss calculation
        if self.loss_type == "huber":
            loss = F.smooth_l1_loss(q_pred, targets)
        else:
            loss = F.mse_loss(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.total_gradient_steps += 1

        # Target network update
        if self.use_target_network:
            if self.target_update == "soft":
                with torch.no_grad():
                    for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            elif self.target_update == "hard":
                if self.total_gradient_steps % self.target_update_every == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = float(loss.item())
        q_mean_val = float(q_pred.mean().item())
        self.losses.append(loss_val)
        self.q_means.append(q_mean_val)

        return loss_val, q_mean_val

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Tuple[float, float]]:
        self.total_env_steps += 1
        self.buffer.add(state, action, reward, next_state, done)

        metrics = None
        if self.total_env_steps >= self.learning_starts and (self.total_env_steps % self.learn_every == 0):
            metrics = self.update()
        return metrics

    def save(self, filepath: str):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_env_steps": self.total_env_steps,
                "total_gradient_steps": self.total_gradient_steps,
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_env_steps = checkpoint.get("total_env_steps", 0)
        self.total_gradient_steps = checkpoint.get("total_gradient_steps", 0)

