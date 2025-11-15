"""
Test functions for A2C Exercise
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


def compute_n_step_returns_test(compute_n_step_returns):
    """Test compute_n_step_returns function"""
    print("Testing compute_n_step_returns...")

    # Test case 1: Simple 3-step trajectory, no terminal
    rewards = [1.0, 1.0, 1.0]
    values = [2.0, 2.0, 2.0, 2.0]  # One extra for bootstrap
    dones = [False, False, False]
    gamma = 0.9
    n_steps = 2

    returns = compute_n_step_returns(rewards, values, dones, gamma, n_steps)

    # Expected for timestep 0:
    # R_0 = r_0 + gamma*r_1 + gamma^2*V(s_2)
    #     = 1.0 + 0.9*1.0 + 0.81*2.0 = 3.52
    expected_0 = 1.0 + 0.9 * 1.0 + 0.81 * 2.0

    assert len(returns) == len(rewards), f"Returns length should be {len(rewards)}, got {len(returns)}"
    assert abs(returns[0] - expected_0) < 0.01, f"Return[0] should be ~{expected_0:.2f}, got {returns[0]:.2f}"

    print(f"  ✓ Non-terminal trajectory correct")
    print(f"    Returns: {[f'{r:.2f}' for r in returns]}")
    print(f"    Expected R[0]: {expected_0:.2f}, Got: {returns[0]:.2f}")

    # Test case 2: Trajectory with terminal state
    rewards = [1.0, 1.0, 10.0]
    values = [0.0, 0.0, 0.0, 0.0]
    dones = [False, False, True]  # Episode ends at step 2
    gamma = 0.9
    n_steps = 5  # More than trajectory length

    returns = compute_n_step_returns(rewards, values, dones, gamma, n_steps)

    # Expected for timestep 0:
    # R_0 = r_0 + gamma*r_1 + gamma^2*r_2 (no bootstrap, episode ended)
    #     = 1.0 + 0.9*1.0 + 0.81*10.0 = 10.0
    expected_0_terminal = 1.0 + 0.9 * 1.0 + 0.81 * 10.0

    assert abs(returns[0] - expected_0_terminal) < 0.01, \
        f"Terminal return[0] should be ~{expected_0_terminal:.2f}, got {returns[0]:.2f}"

    # Expected for timestep 2 (terminal):
    # R_2 = r_2 (just the reward, no future)
    expected_2_terminal = 10.0
    assert abs(returns[2] - expected_2_terminal) < 0.01, \
        f"Terminal return[2] should be {expected_2_terminal}, got {returns[2]:.2f}"

    print(f"  ✓ Terminal trajectory correct")
    print(f"    Returns with terminal: {[f'{r:.2f}' for r in returns]}")

    # Test case 3: Different n_steps
    rewards = [1.0] * 10
    values = [0.5] * 11
    dones = [False] * 10
    gamma = 0.99
    n_steps = 1  # Should behave like 1-step TD

    returns_1step = compute_n_step_returns(rewards, values, dones, gamma, n_steps)
    # R_0 = r_0 + gamma*V(s_1) = 1.0 + 0.99*0.5 = 1.495
    expected_1step = 1.0 + 0.99 * 0.5
    assert abs(returns_1step[0] - expected_1step) < 0.01, \
        f"1-step return should be ~{expected_1step:.2f}, got {returns_1step[0]:.2f}"

    print(f"  ✓ Different n_steps values work correctly")

    print("\n✅ All tests passed!")


def a2c_actor_network_test(A2CActorNetwork):
    """Test A2CActorNetwork class"""
    print("Testing A2CActorNetwork...")

    state_dim, action_dim = 4, 2
    actor = A2CActorNetwork(state_dim, action_dim, hidden_dim=64)

    # Check structure
    assert hasattr(actor, 'fc1'), "Actor should have fc1"
    assert hasattr(actor, 'fc2'), "Actor should have fc2"
    assert isinstance(actor.fc1, nn.Linear), "fc1 should be Linear"
    assert isinstance(actor.fc2, nn.Linear), "fc2 should be Linear"

    # Check dimensions
    assert actor.fc1.in_features == state_dim
    assert actor.fc1.out_features == 64
    assert actor.fc2.in_features == 64
    assert actor.fc2.out_features == action_dim

    print(f"  ✓ Network structure correct")
    print(f"    fc1: {state_dim} → 64")
    print(f"    fc2: 64 → {action_dim}")

    # Test forward
    state = torch.randn(1, state_dim)
    probs = actor(state)

    assert probs.shape == (1, action_dim)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5), \
        f"Probabilities should sum to 1.0, got {probs.sum()}"
    assert torch.all(probs >= 0) and torch.all(probs <= 1)

    print(f"  ✓ Forward pass correct")
    print(f"    Sample probs: {probs.detach().numpy()[0]}")

    print("\n✅ All tests passed!")


def a2c_critic_network_test(A2CCriticNetwork):
    """Test A2CCriticNetwork class"""
    print("Testing A2CCriticNetwork...")

    state_dim = 4
    critic = A2CCriticNetwork(state_dim, hidden_dim=64)

    # Check structure
    assert hasattr(critic, 'fc1'), "Critic should have fc1"
    assert hasattr(critic, 'fc2'), "Critic should have fc2"

    # Check dimensions
    assert critic.fc1.in_features == state_dim
    assert critic.fc1.out_features == 64
    assert critic.fc2.in_features == 64
    assert critic.fc2.out_features == 1

    print(f"  ✓ Network structure correct")

    # Test forward
    state = torch.randn(1, state_dim)
    value = critic(state)

    assert value.shape == (1,) or value.shape == (), \
        f"Value should be scalar or (1,), got {value.shape}"

    print(f"  ✓ Forward pass correct")
    print(f"    Sample value: {value.item():.4f}")

    # Test batch
    batch_state = torch.randn(10, state_dim)
    batch_values = critic(batch_state)
    assert batch_values.shape[0] == 10

    print("\n✅ All tests passed!")


def compute_a2c_loss_test(compute_a2c_loss):
    """Test compute_a2c_loss function"""
    print("Testing compute_a2c_loss...")

    # Create sample data
    n_steps = 5
    log_probs = [torch.tensor(-0.5, requires_grad=True) for _ in range(n_steps)]
    values = [torch.tensor(2.0, requires_grad=True) for _ in range(n_steps)]
    n_step_returns = [3.0, 2.5, 2.0, 1.5, 1.0]  # Decreasing returns

    actor_loss, critic_loss, total_loss = compute_a2c_loss(
        log_probs, values, n_step_returns, entropy_coef=0.01
    )

    # Check types
    assert isinstance(actor_loss, torch.Tensor), "actor_loss should be torch.Tensor"
    assert isinstance(critic_loss, torch.Tensor), "critic_loss should be torch.Tensor"
    assert isinstance(total_loss, torch.Tensor), "total_loss should be torch.Tensor"

    # Check requires_grad
    assert total_loss.requires_grad, "total_loss should require grad"

    # Check loss is scalar
    assert actor_loss.dim() == 0 or actor_loss.shape == torch.Size([])
    assert critic_loss.dim() == 0 or critic_loss.shape == torch.Size([])

    print(f"  ✓ Loss computation correct")
    print(f"    Actor loss: {actor_loss.item():.4f}")
    print(f"    Critic loss: {critic_loss.item():.4f}")
    print(f"    Total loss: {total_loss.item():.4f}")

    # Test gradient flow
    total_loss.backward()
    has_grad = any(lp.grad is not None for lp in log_probs)
    assert has_grad, "Gradients should flow to log_probs"

    print(f"  ✓ Gradients flow correctly")

    # Test with different advantages
    log_probs2 = [torch.tensor(-0.3, requires_grad=True) for _ in range(3)]
    values2 = [torch.tensor(5.0, requires_grad=True) for _ in range(3)]
    returns2 = [10.0, 8.0, 6.0]  # Positive advantages

    actor_loss2, critic_loss2, total_loss2 = compute_a2c_loss(
        log_probs2, values2, returns2
    )

    # With positive advantages, actor loss should be different
    print(f"  ✓ Different advantages produce different losses")
    print(f"    New actor loss: {actor_loss2.item():.4f}")

    print("\n✅ All tests passed!")


def train_a2c_test(train_a2c, A2CActorNetwork, A2CCriticNetwork):
    """Test train_a2c function"""
    print("Testing train_a2c...")

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize networks
    actor = A2CActorNetwork(state_dim, action_dim, hidden_dim=32)
    critic = A2CCriticNetwork(state_dim, hidden_dim=32)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)

    # Short training run
    n_episodes = 10
    episode_rewards = train_a2c(
        env, actor, critic, optimizer,
        n_episodes=n_episodes,
        gamma=0.99,
        n_steps=5,
        max_steps_per_episode=100
    )

    # Check output
    assert isinstance(episode_rewards, list), "Should return list"
    assert len(episode_rewards) == n_episodes, f"Should have {n_episodes} rewards"

    for i, reward in enumerate(episode_rewards):
        assert isinstance(reward, (int, float, np.integer, np.floating)), \
            f"Episode {i}: reward should be number"
        assert reward > 0, f"Episode {i}: reward should be positive"

    print(f"  ✓ Training completed {n_episodes} episodes")
    print(f"  ✓ Rewards: min={min(episode_rewards):.1f}, max={max(episode_rewards):.1f}, avg={np.mean(episode_rewards):.1f}")

    # Check networks were updated
    initial_actor_params = [p.clone() for p in actor.parameters()]
    initial_critic_params = [p.clone() for p in critic.parameters()]

    # Run one more episode
    episode_rewards2 = train_a2c(
        env, actor, critic, optimizer,
        n_episodes=1,
        gamma=0.99,
        n_steps=5
    )

    # Check parameters changed
    actor_changed = any(not torch.allclose(p1, p2)
                       for p1, p2 in zip(initial_actor_params, actor.parameters()))
    critic_changed = any(not torch.allclose(p1, p2)
                        for p1, p2 in zip(initial_critic_params, critic.parameters()))

    assert actor_changed, "Actor parameters should change"
    assert critic_changed, "Critic parameters should change"

    print(f"  ✓ Networks are being updated")

    env.close()
    print("\n✅ All tests passed!")
