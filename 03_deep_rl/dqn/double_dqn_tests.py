"""
Test functions for Double DQN Exercise
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import namedtuple


def qnetwork_test(QNetwork):
    """Test QNetwork class"""
    print("Testing QNetwork...")

    state_dim, action_dim = 4, 2
    q_net = QNetwork(state_dim, action_dim, hidden_dim=64)

    # Check network structure
    assert hasattr(q_net, 'fc1'), "Network should have fc1 layer"
    assert hasattr(q_net, 'fc2'), "Network should have fc2 layer"
    assert hasattr(q_net, 'fc3'), "Network should have fc3 layer"
    assert isinstance(q_net.fc1, nn.Linear), "fc1 should be Linear"
    assert isinstance(q_net.fc2, nn.Linear), "fc2 should be Linear"
    assert isinstance(q_net.fc3, nn.Linear), "fc3 should be Linear"

    # Check dimensions
    assert q_net.fc1.in_features == state_dim
    assert q_net.fc1.out_features == 64
    assert q_net.fc2.in_features == 64
    assert q_net.fc2.out_features == 64
    assert q_net.fc3.in_features == 64
    assert q_net.fc3.out_features == action_dim

    print(f"  ✓ Network structure correct")
    print(f"    fc1: {state_dim} → 64")
    print(f"    fc2: 64 → 64")
    print(f"    fc3: 64 → {action_dim}")

    # Test forward pass
    state = torch.randn(1, state_dim)
    q_values = q_net(state)

    assert q_values.shape == (1, action_dim), f"Expected shape (1, {action_dim}), got {q_values.shape}"
    assert not torch.isnan(q_values).any(), "Q-values contain NaN"

    print(f"  ✓ Forward pass correct")
    print(f"    Output shape: {q_values.shape}")
    print(f"    Sample Q-values: {q_values.detach().numpy()[0]}")

    # Test with batch
    batch_state = torch.randn(32, state_dim)
    batch_q = q_net(batch_state)
    assert batch_q.shape == (32, action_dim)

    print("\n✅ All tests passed!")


def replay_buffer_test(ReplayBuffer):
    """Test ReplayBuffer class"""
    print("Testing ReplayBuffer...")

    buffer = ReplayBuffer(capacity=100)

    # Test initial state
    assert len(buffer) == 0, "Buffer should start empty"

    # Test push
    for i in range(50):
        buffer.push(
            state=np.array([i, i+1, i+2, i+3]),
            action=i % 2,
            reward=float(i),
            next_state=np.array([i+1, i+2, i+3, i+4]),
            done=False
        )

    assert len(buffer) == 50, f"Buffer should have 50 items, got {len(buffer)}"

    print(f"  ✓ Push works correctly ({len(buffer)} items)")

    # Test sample
    batch = buffer.sample(batch_size=10)
    assert len(batch) == 10, f"Batch should have 10 items, got {len(batch)}"
    assert hasattr(batch[0], 'state'), "Transition should have state"
    assert hasattr(batch[0], 'action'), "Transition should have action"
    assert hasattr(batch[0], 'reward'), "Transition should have reward"
    assert hasattr(batch[0], 'next_state'), "Transition should have next_state"
    assert hasattr(batch[0], 'done'), "Transition should have done"

    print(f"  ✓ Sample works correctly")
    print(f"    Sample transition: state={batch[0].state}, action={batch[0].action}, reward={batch[0].reward}")

    # Test capacity limit
    for i in range(100):
        buffer.push(
            state=np.array([0, 0, 0, 0]),
            action=0,
            reward=0.0,
            next_state=np.array([0, 0, 0, 0]),
            done=False
        )

    assert len(buffer) == 100, f"Buffer should be capped at 100, got {len(buffer)}"
    print(f"  ✓ Capacity limit works ({len(buffer)} items)")

    print("\n✅ All tests passed!")


def compute_double_dqn_loss_test(compute_double_dqn_loss, QNetwork, ReplayBuffer):
    """Test compute_double_dqn_loss function"""
    print("Testing compute_double_dqn_loss...")

    state_dim, action_dim = 4, 2

    # Create networks
    online_net = QNetwork(state_dim, action_dim, hidden_dim=32)
    target_net = QNetwork(state_dim, action_dim, hidden_dim=32)
    target_net.load_state_dict(online_net.state_dict())

    # Create sample batch
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
    batch = [
        Transition(
            state=np.random.randn(state_dim),
            action=np.random.randint(0, action_dim),
            reward=np.random.rand(),
            next_state=np.random.randn(state_dim),
            done=False
        )
        for _ in range(32)
    ]

    # Compute loss
    loss = compute_double_dqn_loss(batch, online_net, target_net, gamma=0.99)

    # Check loss properties
    assert isinstance(loss, torch.Tensor), "Loss should be torch.Tensor"
    assert loss.dim() == 0 or loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.requires_grad, "Loss should require grad"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    print(f"  ✓ Loss computation correct")
    print(f"    Loss value: {loss.item():.6f}")

    # Test with terminal states
    terminal_batch = [
        Transition(
            state=np.random.randn(state_dim),
            action=np.random.randint(0, action_dim),
            reward=1.0,
            next_state=np.random.randn(state_dim),
            done=True
        )
        for _ in range(16)
    ]

    loss_terminal = compute_double_dqn_loss(terminal_batch, online_net, target_net, gamma=0.99)
    assert loss_terminal.item() >= 0, "Loss with terminal states should be non-negative"

    print(f"  ✓ Terminal state handling correct")
    print(f"    Loss value (terminal): {loss_terminal.item():.6f}")

    # Test gradient flow
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in online_net.parameters())
    assert has_grad, "At least some online_net parameters should have gradients"

    print(f"  ✓ Gradients flow correctly")

    print("\n✅ All tests passed!")


def update_target_network_test(update_target_network, QNetwork):
    """Test update_target_network function"""
    print("Testing update_target_network...")

    state_dim, action_dim = 4, 2

    # Create networks with different weights
    online_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)

    # Make sure they start different
    with torch.no_grad():
        for p in online_net.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Check they are different before update
    params_different = False
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        if not torch.allclose(p_online, p_target):
            params_different = True
            break

    assert params_different, "Networks should start with different parameters"
    print(f"  ✓ Networks start with different parameters")

    # Update target network
    update_target_network(online_net, target_net)

    # Check they are now the same
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target), \
            "Target network parameters should match online network after update"

    print(f"  ✓ Target network updated correctly")

    # Test that changing online doesn't affect target now
    with torch.no_grad():
        for p in online_net.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    params_different_after = False
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        if not torch.allclose(p_online, p_target):
            params_different_after = True
            break

    assert params_different_after, "Networks should be different after modifying online"
    print(f"  ✓ Networks are independent after update")

    print("\n✅ All tests passed!")


def train_double_dqn_test(train_double_dqn):
    """Test train_double_dqn function"""
    print("Testing train_double_dqn...")

    env = gym.make('CartPole-v1')

    # Short training run
    n_episodes = 10
    episode_rewards, trained_net = train_double_dqn(
        env,
        n_episodes=n_episodes,
        gamma=0.99,
        epsilon_decay=0.95,
        batch_size=32,
        target_update_freq=5
    )

    # Check outputs
    assert isinstance(episode_rewards, list), "Should return list of rewards"
    assert len(episode_rewards) == n_episodes, f"Should have {n_episodes} rewards, got {len(episode_rewards)}"
    assert isinstance(trained_net, nn.Module), "Should return trained network"

    # Check rewards are valid
    for i, reward in enumerate(episode_rewards):
        assert isinstance(reward, (int, float, np.integer, np.floating)), \
            f"Episode {i}: reward should be number"
        assert reward > 0, f"Episode {i}: reward should be positive"

    print(f"  ✓ Training completed {n_episodes} episodes")
    print(f"  ✓ Rewards: min={min(episode_rewards):.1f}, max={max(episode_rewards):.1f}, avg={np.mean(episode_rewards):.1f}")

    # Check network was trained (parameters changed)
    # Run one more episode to verify
    initial_params = [p.clone() for p in trained_net.parameters()]

    episode_rewards2, trained_net2 = train_double_dqn(
        env,
        n_episodes=1,
        gamma=0.99,
        batch_size=32
    )

    params_changed = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(initial_params, trained_net2.parameters())
    )

    assert params_changed, "Network parameters should change during training"
    print(f"  ✓ Network is being updated")

    # Test network can make predictions
    state = torch.FloatTensor(env.reset()[0])
    with torch.no_grad():
        q_values = trained_net(state.unsqueeze(0))

    assert q_values.shape[1] == env.action_space.n, "Network should output correct number of Q-values"
    print(f"  ✓ Trained network makes valid predictions")
    print(f"    Sample Q-values: {q_values.numpy()[0]}")

    env.close()
    print("\n✅ All tests passed!")
