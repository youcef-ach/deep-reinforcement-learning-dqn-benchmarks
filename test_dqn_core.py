import unittest
import numpy as np
import torch
import torch.nn as nn
from dqn_core import (
    ReplayBuffer,
    MLPQNetwork,
    NatureCNNQNetwork,
    MLPDuelingQNetwork,
    NatureCNNDuelingQNetwork,
    DQNAgent,
    build_q_network,
)


class TestDQNCore(unittest.TestCase):
    def test_nature_cnn_parameter_count(self):
        """Verify Nature CNN matches exact architecture and parameter count (1,687,206)."""
        cnn = NatureCNNQNetwork(in_channels=4, action_dim=6)
        total_params = sum(p.numel() for p in cnn.parameters())
        self.assertEqual(total_params, 1687206)

        # Test forward pass with uint8 input
        dummy_obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
        out = cnn(dummy_obs)
        self.assertEqual(out.shape, (2, 6))

    def test_dueling_aggregation_formula(self):
        """Verify dueling formula Q(s, a) = V(s) + A(s, a) - mean(A(s, :))."""
        mlp_dueling = MLPDuelingQNetwork(state_dim=4, action_dim=2, hidden_dim=64)
        x = torch.randn(5, 4)
        q = mlp_dueling(x)
        self.assertEqual(q.shape, (5, 2))

        # Check manual computation matches module forward
        feat = mlp_dueling.shared(x)
        val = mlp_dueling.value_stream(feat)
        adv = mlp_dueling.advantage_stream(feat)
        expected_q = val + (adv - adv.mean(dim=-1, keepdim=True))
        self.assertTrue(torch.allclose(q, expected_q, atol=1e-6))

        # Also verify for CNN dueling
        cnn_dueling = NatureCNNDuelingQNetwork(in_channels=4, action_dim=6)
        x_img = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
        q_img = cnn_dueling(x_img)
        self.assertEqual(q_img.shape, (2, 6))

    def test_target_computations(self):
        """Verify target calculations for standard DQN, Double DQN, and no target network."""
        # Setup agent
        agent = DQNAgent(
            state_shape=4,
            action_dim=2,
            gamma=0.99,
            learning_starts=10,
            seed=42,
        )

        rewards = torch.tensor([1.0, 2.0, 3.0])
        next_states = torch.randn(3, 4)
        dones = torch.tensor([0.0, 1.0, 0.0])  # middle transition is terminal

        # 1. Standard DQN
        agent.double_dqn = False
        agent.use_target_network = True
        targets_std = agent.compute_targets(rewards, next_states, dones)
        with torch.no_grad():
            expected_std = rewards + 0.99 * (1.0 - dones) * agent.target_net(next_states).max(dim=-1)[0]
        self.assertTrue(torch.allclose(targets_std, expected_std, atol=1e-6))
        # Terminal step check
        self.assertAlmostEqual(targets_std[1].item(), 2.0, places=5)

        # 2. Double DQN
        agent.double_dqn = True
        agent.use_target_network = True
        targets_ddqn = agent.compute_targets(rewards, next_states, dones)
        with torch.no_grad():
            best_a = agent.q_net(next_states).argmax(dim=-1, keepdim=True)
            ddqn_val = agent.target_net(next_states).gather(dim=-1, index=best_a).squeeze(-1)
            expected_ddqn = rewards + 0.99 * (1.0 - dones) * ddqn_val
        self.assertTrue(torch.allclose(targets_ddqn, expected_ddqn, atol=1e-6))

        # 3. No target network
        agent.double_dqn = False
        agent.use_target_network = False
        targets_notarget = agent.compute_targets(rewards, next_states, dones)
        with torch.no_grad():
            expected_notarget = rewards + 0.99 * (1.0 - dones) * agent.q_net(next_states).max(dim=-1)[0]
        self.assertTrue(torch.allclose(targets_notarget, expected_notarget, atol=1e-6))

    def test_soft_and_hard_updates(self):
        """Verify soft update equation theta_target = tau * theta + (1 - tau) * theta_target."""
        agent_soft = DQNAgent(
            state_shape=4,
            action_dim=2,
            target_update="soft",
            tau=0.01,
            batch_size=4,
            learning_starts=4,
        )
        # Populate buffer
        for _ in range(4):
            agent_soft.buffer.add(np.zeros(4), 0, 1.0, np.zeros(4), False)

        p_online_before = next(agent_soft.q_net.parameters()).clone()
        p_target_before = next(agent_soft.target_net.parameters()).clone()

        agent_soft.update()

        p_online_after = next(agent_soft.q_net.parameters())
        p_target_after = next(agent_soft.target_net.parameters())

        expected_target = 0.01 * p_online_after.data + 0.99 * p_target_before.data
        self.assertTrue(torch.allclose(p_target_after, expected_target, atol=1e-5))

        # Test hard update
        agent_hard = DQNAgent(
            state_shape=4,
            action_dim=2,
            target_update="hard",
            target_update_every=5,
            batch_size=4,
            learning_starts=4,
        )
        for _ in range(4):
            agent_hard.buffer.add(np.zeros(4), 0, 1.0, np.zeros(4), False)

        init_target_p = next(agent_hard.target_net.parameters()).clone()
        for i in range(4):
            agent_hard.update()
        # After 4 updates, target should still NOT be updated
        self.assertTrue(torch.allclose(next(agent_hard.target_net.parameters()), init_target_p))
        # 5th update triggers hard copy
        agent_hard.update()
        self.assertTrue(torch.allclose(next(agent_hard.target_net.parameters()), next(agent_hard.q_net.parameters())))

    def test_replay_buffer_variants(self):
        """Verify standard replay buffer vs no experience replay."""
        # Standard buffer
        buf = ReplayBuffer(capacity=100, state_dim=4, batch_size=8, use_replay=True)
        for i in range(20):
            buf.add(np.full(4, i), i % 2, float(i), np.full(4, i + 1), False)
        self.assertEqual(len(buf), 20)
        s, a, r, ns, d = buf.sample()
        self.assertEqual(s.shape, (8, 4))
        self.assertEqual(a.shape, (8,))

        # No replay buffer (memory of size batch_size)
        buf_no_replay = ReplayBuffer(capacity=100, state_dim=4, batch_size=5, use_replay=False)
        self.assertEqual(buf_no_replay.capacity, 5)
        for i in range(7):
            buf_no_replay.add(np.full(4, i), i % 2, float(i), np.full(4, i + 1), False)
        self.assertEqual(len(buf_no_replay), 5)
        s, a, r, ns, d = buf_no_replay.sample()
        # Most recent consecutive 5 transitions are 2, 3, 4, 5, 6
        self.assertTrue(np.allclose(r.numpy(), [2.0, 3.0, 4.0, 5.0, 6.0]))


if __name__ == "__main__":
    unittest.main()

