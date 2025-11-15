"""
Test functions for Actor-Critic Exercise
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


def actor_network_test(ActorNetwork):
    """Test ActorNetwork class"""
    print("Testing ActorNetwork...")

    state_dim, action_dim = 4, 2
    actor = ActorNetwork(state_dim, action_dim, hidden_dim=64)

    # Check network structure
    assert hasattr(actor, 'fc1'), "Actor should have fc1 layer"
    assert hasattr(actor, 'fc2'), "Actor should have fc2 layer"
    assert isinstance(actor.fc1, nn.Linear), "fc1 should be Linear layer"
    assert isinstance(actor.fc2, nn.Linear), "fc2 should be Linear layer"

    # Check dimensions
    assert actor.fc1.in_features == state_dim, f"fc1 input should be {state_dim}"
    assert actor.fc1.out_features == 64, "fc1 output should be 64"
    assert actor.fc2.in_features == 64, "fc2 input should be 64"
    assert actor.fc2.out_features == action_dim, f"fc2 output should be {action_dim}"

    print(f"  ✓ Network structure correct")
    print(f"    fc1: {state_dim} → 64")
    print(f"    fc2: 64 → {action_dim}")

    # Test forward pass
    state = torch.randn(1, state_dim)
    action_probs = actor(state)

    # Check output
    assert action_probs.shape == (1, action_dim), f"Expected shape (1, {action_dim}), got {action_probs.shape}"
    assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-5), \
        f"Probabilities should sum to 1.0, got {action_probs.sum()}"
    assert torch.all(action_probs >= 0) and torch.all(action_probs <= 1), \
        "Probabilities should be in [0, 1]"

    print(f"  ✓ Forward pass correct")
    print(f"    Output shape: {action_probs.shape}")
    print(f"    Sample probs: {action_probs.detach().numpy()[0]}")
    print(f"    Sum: {action_probs.sum().item():.6f}")

    # Test with batch
    batch_state = torch.randn(10, state_dim)
    batch_probs = actor(batch_state)
    assert batch_probs.shape == (10, action_dim)

    print("\n✅ All tests passed!")


def critic_network_test(CriticNetwork):
    """Test CriticNetwork class"""
    print("Testing CriticNetwork...")

    state_dim = 4
    critic = CriticNetwork(state_dim, hidden_dim=64)

    # Check network structure
    assert hasattr(critic, 'fc1'), "Critic should have fc1 layer"
    assert hasattr(critic, 'fc2'), "Critic should have fc2 layer"
    assert isinstance(critic.fc1, nn.Linear), "fc1 should be Linear layer"
    assert isinstance(critic.fc2, nn.Linear), "fc2 should be Linear layer"

    # Check dimensions
    assert critic.fc1.in_features == state_dim, f"fc1 input should be {state_dim}"
    assert critic.fc1.out_features == 64, "fc1 output should be 64"
    assert critic.fc2.in_features == 64, "fc2 input should be 64"
    assert critic.fc2.out_features == 1, "fc2 output should be 1 (value)"

    print(f"  ✓ Network structure correct")
    print(f"    fc1: {state_dim} → 64")
    print(f"    fc2: 64 → 1")

    # Test forward pass
    state = torch.randn(1, state_dim)
    value = critic(state)

    # Check output
    assert value.shape == (1,) or value.shape == (), \
        f"Expected scalar or shape (1,), got {value.shape}"
    assert isinstance(value.item(), float), "Value should be a float"

    print(f"  ✓ Forward pass correct")
    print(f"    Output shape: {value.shape}")
    print(f"    Sample value: {value.item():.4f}")

    # Test with batch
    batch_state = torch.randn(10, state_dim)
    batch_values = critic(batch_state)
    assert batch_values.shape == (10,) or batch_values.shape == (10, 1)

    print(f"  ✓ Batch processing works")

    print("\n✅ All tests passed!")


def select_action_test(select_action, ActorNetwork):
    """Test select_action function"""
    print("Testing select_action...")

    state_dim, action_dim = 4, 2
    actor = ActorNetwork(state_dim, action_dim)

    # Test with numpy state
    state = np.random.randn(state_dim)
    action, log_prob = select_action(actor, state)

    # Check action
    assert isinstance(action, (int, np.integer)), f"Action should be int, got {type(action)}"
    assert 0 <= action < action_dim, f"Action {action} out of range [0, {action_dim})"

    # Check log_prob
    assert isinstance(log_prob, torch.Tensor), "log_prob should be torch.Tensor"
    assert log_prob.requires_grad, "log_prob should require grad for backprop"
    assert log_prob.dim() == 0 or log_prob.shape == (1,), "log_prob should be scalar"
    assert log_prob.item() <= 0, f"log probability should be <= 0, got {log_prob.item()}"

    print(f"  ✓ Basic functionality correct")
    print(f"    Action: {action}")
    print(f"    Log prob: {log_prob.item():.4f}")

    # Test multiple calls (should get different actions sometimes)
    actions = []
    for _ in range(100):
        action, _ = select_action(actor, state)
        actions.append(action)

    unique_actions = len(set(actions))
    print(f"  ✓ Stochasticity check: {unique_actions} unique actions in 100 samples")
    assert unique_actions > 1, "Should sample different actions (stochastic policy)"

    print("\n✅ All tests passed!")


def compute_ac_loss_test(compute_ac_loss):
    """Test compute_ac_loss function"""
    print("Testing compute_ac_loss...")

    # Test case 1: Non-terminal transition
    log_prob = torch.tensor(-0.5, requires_grad=True)
    value = torch.tensor(10.0, requires_grad=True)
    next_value = torch.tensor(8.0, requires_grad=True)
    reward = 1.0
    done = False
    gamma = 0.99

    actor_loss, critic_loss = compute_ac_loss(log_prob, value, next_value, reward, done, gamma)

    # Check types
    assert isinstance(actor_loss, torch.Tensor), "actor_loss should be torch.Tensor"
    assert isinstance(critic_loss, torch.Tensor), "critic_loss should be torch.Tensor"

    # Check gradients
    assert actor_loss.requires_grad, "actor_loss should require grad"
    assert critic_loss.requires_grad, "critic_loss should require grad"

    # TD target = reward + gamma * next_value = 1.0 + 0.99 * 8.0 = 8.92
    # Advantage = 8.92 - 10.0 = -1.08
    # Actor loss = -(-0.5) * (-1.08) = -0.54
    # Critic loss = (-1.08)^2 = 1.1664

    expected_td_target = reward + gamma * next_value.item()
    expected_advantage = expected_td_target - value.item()
    expected_actor_loss = -(log_prob.item() * expected_advantage)
    expected_critic_loss = expected_advantage ** 2

    print(f"  Non-terminal transition:")
    print(f"    TD target: {expected_td_target:.4f}")
    print(f"    Advantage (TD error): {expected_advantage:.4f}")
    print(f"    Actor loss: {actor_loss.item():.4f} (expected ~{expected_actor_loss:.4f})")
    print(f"    Critic loss: {critic_loss.item():.4f} (expected ~{expected_critic_loss:.4f})")

    assert abs(actor_loss.item() - expected_actor_loss) < 0.1, \
        f"Actor loss mismatch: {actor_loss.item()} vs {expected_actor_loss}"

    # Test case 2: Terminal transition
    log_prob2 = torch.tensor(-0.3, requires_grad=True)
    value2 = torch.tensor(5.0, requires_grad=True)
    next_value2 = torch.tensor(0.0, requires_grad=True)  # Ignored for terminal
    reward2 = 10.0
    done2 = True

    actor_loss2, critic_loss2 = compute_ac_loss(log_prob2, value2, next_value2, reward2, done2, gamma)

    # TD target = reward = 10.0 (no next_value for terminal)
    # Advantage = 10.0 - 5.0 = 5.0
    # Actor loss = -(-0.3) * 5.0 = 1.5
    # Critic loss = 5.0^2 = 25.0

    expected_advantage2 = reward2 - value2.item()
    expected_actor_loss2 = -(log_prob2.item() * expected_advantage2)

    print(f"\n  Terminal transition:")
    print(f"    TD target: {reward2:.4f}")
    print(f"    Advantage: {expected_advantage2:.4f}")
    print(f"    Actor loss: {actor_loss2.item():.4f} (expected ~{expected_actor_loss2:.4f})")

    assert abs(actor_loss2.item() - expected_actor_loss2) < 0.1, \
        "Actor loss incorrect for terminal state"

    print("\n✅ All tests passed!")


def train_actor_critic_test(train_actor_critic, ActorNetwork, CriticNetwork, select_action):
    """Test train_actor_critic function"""
    print("Testing train_actor_critic...")

    # Create simple environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize networks
    actor = ActorNetwork(state_dim, action_dim, hidden_dim=32)
    critic = CriticNetwork(state_dim, hidden_dim=32)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    # Short training run
    n_episodes = 10
    episode_rewards = train_actor_critic(
        env, actor, critic, actor_optimizer, critic_optimizer,
        n_episodes=n_episodes, gamma=0.99, max_steps=100
    )

    # Check output
    assert isinstance(episode_rewards, list), "Should return list of rewards"
    assert len(episode_rewards) == n_episodes, f"Should have {n_episodes} rewards, got {len(episode_rewards)}"

    # Check all rewards are numbers
    for i, reward in enumerate(episode_rewards):
        assert isinstance(reward, (int, float, np.integer, np.floating)), \
            f"Episode {i}: reward should be number, got {type(reward)}"
        assert reward > 0, f"Episode {i}: reward should be positive, got {reward}"

    print(f"  ✓ Training completed {n_episodes} episodes")
    print(f"  ✓ Rewards: min={min(episode_rewards):.1f}, max={max(episode_rewards):.1f}, avg={np.mean(episode_rewards):.1f}")

    # Check that networks were updated
    initial_actor_params = [p.clone() for p in actor.parameters()]
    initial_critic_params = [p.clone() for p in critic.parameters()]

    # Run one more episode
    episode_rewards2 = train_actor_critic(
        env, actor, critic, actor_optimizer, critic_optimizer,
        n_episodes=1, gamma=0.99, max_steps=100
    )

    # At least one parameter should have changed
    actor_changed = any(not torch.allclose(p1, p2) for p1, p2 in zip(initial_actor_params, actor.parameters()))
    critic_changed = any(not torch.allclose(p1, p2) for p1, p2 in zip(initial_critic_params, critic.parameters()))

    assert actor_changed, "Actor parameters should be updated"
    assert critic_changed, "Critic parameters should be updated"

    print(f"  ✓ Networks are being updated")

    env.close()
    print("\n✅ All tests passed!")
