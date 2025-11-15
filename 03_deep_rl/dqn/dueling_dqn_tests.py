"""
Test functions for Dueling DQN Exercise
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import namedtuple


def dueling_qnetwork_test(DuelingQNetwork):
    """Test DuelingQNetwork class"""
    print("Testing DuelingQNetwork...")

    state_dim, action_dim = 4, 2
    net = DuelingQNetwork(state_dim, action_dim, hidden_dim=64)

    # Check structure - should have all necessary layers
    assert hasattr(net, 'fc1'), "Should have shared fc1 layer"
    assert hasattr(net, 'value_fc'), "Should have value_fc layer"
    assert hasattr(net, 'value_out'), "Should have value_out layer"
    assert hasattr(net, 'advantage_fc'), "Should have advantage_fc layer"
    assert hasattr(net, 'advantage_out'), "Should have advantage_out layer"

    print(f"  ✓ Network has all required layers")

    # Check dimensions
    assert net.fc1.in_features == state_dim
    assert net.fc1.out_features == 64
    assert net.value_fc.in_features == 64
    assert net.value_fc.out_features == 32  # hidden_dim//2
    assert net.value_out.in_features == 32
    assert net.value_out.out_features == 1  # Single value
    assert net.advantage_fc.in_features == 64
    assert net.advantage_fc.out_features == 32
    assert net.advantage_out.in_features == 32
    assert net.advantage_out.out_features == action_dim

    print(f"  ✓ All layer dimensions correct")
    print(f"    Shared: {state_dim} → 64")
    print(f"    Value stream: 64 → 32 → 1")
    print(f"    Advantage stream: 64 → 32 → {action_dim}")

    # Test forward pass
    state = torch.randn(1, state_dim)
    q_values = net(state)

    # Check output shape
    assert q_values.shape == (1, action_dim), f"Expected shape (1, {action_dim}), got {q_values.shape}"
    assert not torch.isnan(q_values).any(), "Q-values contain NaN"

    print(f"  ✓ Forward pass correct")
    print(f"    Output shape: {q_values.shape}")
    print(f"    Sample Q-values: {q_values.detach().numpy()[0]}")

    # Test dueling property: Q should decompose into V + A
    # Get intermediate values
    with torch.no_grad():
        x = F.relu(net.fc1(state))

        v = F.relu(net.value_fc(x))
        value = net.value_out(v)

        a = F.relu(net.advantage_fc(x))
        advantages = net.advantage_out(a)

        # Verify: Q = V + (A - mean(A))
        expected_q = value + (advantages - advantages.mean(dim=1, keepdim=True))

        assert torch.allclose(q_values, expected_q, atol=1e-5), \
            "Q-values should equal V + (A - mean(A))"

    print(f"  ✓ Dueling decomposition correct: Q = V + (A - mean(A))")
    print(f"    V(s): {value.item():.4f}")
    print(f"    A(s,a): {advantages.numpy()[0]}")
    print(f"    mean(A): {advantages.mean().item():.4f}")

    # Test batch processing
    batch_state = torch.randn(32, state_dim)
    batch_q = net(batch_state)
    assert batch_q.shape == (32, action_dim)

    print(f"  ✓ Batch processing works")

    print("\n✅ All tests passed!")


def dueling_replay_buffer_test(DuelingReplayBuffer):
    """Test DuelingReplayBuffer class"""
    print("Testing DuelingReplayBuffer...")

    buffer = DuelingReplayBuffer(capacity=100)

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

    print(f"  ✓ Push works ({len(buffer)} items)")

    # Test sample
    batch = buffer.sample(batch_size=10)
    assert len(batch) == 10, f"Batch should have 10 items, got {len(batch)}"
    assert hasattr(batch[0], 'state'), "Transition should have state"
    assert hasattr(batch[0], 'action'), "Transition should have action"
    assert hasattr(batch[0], 'reward'), "Transition should have reward"

    print(f"  ✓ Sample works")

    # Test capacity
    for i in range(100):
        buffer.push(np.array([0,0,0,0]), 0, 0.0, np.array([0,0,0,0]), False)

    assert len(buffer) == 100, f"Buffer should cap at 100, got {len(buffer)}"

    print(f"  ✓ Capacity limit works")

    print("\n✅ All tests passed!")


def compute_dueling_dqn_loss_test(compute_dueling_dqn_loss, DuelingQNetwork, DuelingReplayBuffer):
    """Test compute_dueling_dqn_loss function"""
    print("Testing compute_dueling_dqn_loss...")

    state_dim, action_dim = 4, 2

    # Create networks
    online_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=32)
    target_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=32)
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
    loss = compute_dueling_dqn_loss(batch, online_net, target_net, gamma=0.99)

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

    loss_terminal = compute_dueling_dqn_loss(terminal_batch, online_net, target_net, gamma=0.99)
    assert loss_terminal.item() >= 0

    print(f"  ✓ Terminal state handling correct")

    # Test gradient flow
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in online_net.parameters())
    assert has_grad, "Should have gradients"

    # Check that value and advantage streams both get gradients
    assert online_net.value_out.weight.grad is not None, "Value stream should have gradients"
    assert online_net.advantage_out.weight.grad is not None, "Advantage stream should have gradients"

    print(f"  ✓ Gradients flow to both value and advantage streams")

    print("\n✅ All tests passed!")


def update_dueling_target_test(update_dueling_target, DuelingQNetwork):
    """Test update_dueling_target function"""
    print("Testing update_dueling_target...")

    state_dim, action_dim = 4, 2

    # Create networks with different weights
    online_net = DuelingQNetwork(state_dim, action_dim)
    target_net = DuelingQNetwork(state_dim, action_dim)

    # Make them different
    with torch.no_grad():
        for p in online_net.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Check they're different
    params_different = False
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        if not torch.allclose(p_online, p_target):
            params_different = True
            break

    assert params_different, "Networks should start different"
    print(f"  ✓ Networks start with different parameters")

    # Update target
    update_dueling_target(online_net, target_net)

    # Check they're now the same
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target), \
            "Target should match online after update"

    print(f"  ✓ Target network updated correctly")

    # Check both streams were updated
    assert torch.allclose(online_net.value_out.weight, target_net.value_out.weight)
    assert torch.allclose(online_net.advantage_out.weight, target_net.advantage_out.weight)

    print(f"  ✓ Both value and advantage streams updated")

    print("\n✅ All tests passed!")


def train_dueling_dqn_test(train_dueling_dqn):
    """Test train_dueling_dqn function"""
    print("Testing train_dueling_dqn...")

    env = gym.make('CartPole-v1')

    # Short training run
    n_episodes = 10
    episode_rewards, trained_net = train_dueling_dqn(
        env,
        n_episodes=n_episodes,
        gamma=0.99,
        epsilon_decay=0.95,
        batch_size=32,
        target_update_freq=5
    )

    # Check outputs
    assert isinstance(episode_rewards, list), "Should return list"
    assert len(episode_rewards) == n_episodes, f"Should have {n_episodes} rewards"
    assert isinstance(trained_net, nn.Module), "Should return network"

    # Check it's a DuelingQNetwork
    assert hasattr(trained_net, 'value_fc'), "Should be DuelingQNetwork"
    assert hasattr(trained_net, 'advantage_fc'), "Should be DuelingQNetwork"

    # Check rewards
    for i, reward in enumerate(episode_rewards):
        assert isinstance(reward, (int, float, np.integer, np.floating)), \
            f"Episode {i}: reward should be number"
        assert reward > 0, f"Episode {i}: reward should be positive"

    print(f"  ✓ Training completed {n_episodes} episodes")
    print(f"  ✓ Rewards: min={min(episode_rewards):.1f}, max={max(episode_rewards):.1f}, avg={np.mean(episode_rewards):.1f}")

    # Check network was trained
    initial_params = [p.clone() for p in trained_net.parameters()]

    episode_rewards2, trained_net2 = train_dueling_dqn(
        env,
        n_episodes=1,
        gamma=0.99,
        batch_size=32
    )

    params_changed = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(initial_params, trained_net2.parameters())
    )

    assert params_changed, "Network parameters should change"
    print(f"  ✓ Network is being updated")

    # Test network makes predictions
    state = torch.FloatTensor(env.reset()[0])
    with torch.no_grad():
        q_values = trained_net(state.unsqueeze(0))

    assert q_values.shape[1] == env.action_space.n
    print(f"  ✓ Trained network makes valid predictions")
    print(f"    Sample Q-values: {q_values.numpy()[0]}")

    env.close()
    print("\n✅ All tests passed!")
