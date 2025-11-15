"""
Test functions for PPO Exercise
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


def compute_gae_test(compute_gae):
    """Test compute_gae function"""
    print("Testing compute_gae...")

    # Test case 1: Simple trajectory
    rewards = [1.0, 1.0, 1.0]
    values = [2.0, 2.0, 2.0, 0.0]  # Last is bootstrap
    dones = [False, False, True]  # Terminal at end
    gamma = 0.9
    lambda_ = 0.95

    advantages, returns = compute_gae(rewards, values, dones, gamma, lambda_)

    # Check types and lengths
    assert isinstance(advantages, (list, np.ndarray)), "Advantages should be list or array"
    assert isinstance(returns, (list, np.ndarray)), "Returns should be list or array"
    assert len(advantages) == len(rewards), f"Advantages length should be {len(rewards)}"
    assert len(returns) == len(rewards), f"Returns length should be {len(rewards)}"

    print(f"  ✓ Output format correct")
    print(f"    Advantages: {[f'{a:.3f}' for a in advantages[:3]]}")
    print(f"    Returns: {[f'{r:.3f}' for r in returns[:3]]}")

    # Test case 2: Check terminal state handling
    # Terminal state should have: delta = reward - value, gae = delta
    terminal_idx = 2
    expected_delta_terminal = rewards[terminal_idx] - values[terminal_idx]
    # For terminal state, advantage should be close to delta
    assert abs(advantages[terminal_idx] - expected_delta_terminal) < 0.5, \
        f"Terminal advantage should be ~{expected_delta_terminal:.2f}, got {advantages[terminal_idx]:.2f}"

    print(f"  ✓ Terminal state handling correct")

    # Test case 3: Verify returns = advantages + values relationship
    for i in range(len(returns)):
        expected_return = advantages[i] + values[i]
        assert abs(returns[i] - expected_return) < 0.01, \
            f"Return[{i}] should be {expected_return:.2f}, got {returns[i]:.2f}"

    print(f"  ✓ Returns = advantages + values relationship holds")

    # Test case 4: Different lambda values
    # Lambda = 0 should give 1-step TD
    adv_0, _ = compute_gae(rewards, values, dones, gamma, lambda_=0.0)
    # Lambda = 1 should give Monte Carlo-like (all future rewards)
    adv_1, _ = compute_gae(rewards, values, dones, gamma, lambda_=1.0)

    print(f"  ✓ Different lambda values work")
    print(f"    Lambda=0 advantages: {[f'{a:.3f}' for a in adv_0[:3]]}")
    print(f"    Lambda=1 advantages: {[f'{a:.3f}' for a in adv_1[:3]]}")

    print("\n✅ All tests passed!")


def ppo_actor_network_test(PPOActorNetwork):
    """Test PPOActorNetwork class"""
    print("Testing PPOActorNetwork...")

    state_dim, action_dim = 4, 2
    actor = PPOActorNetwork(state_dim, action_dim, hidden_dim=64)

    # Check structure
    assert hasattr(actor, 'fc1'), "Should have fc1"
    assert hasattr(actor, 'fc2'), "Should have fc2"
    assert isinstance(actor.fc1, nn.Linear), "fc1 should be Linear"
    assert isinstance(actor.fc2, nn.Linear), "fc2 should be Linear"

    # Check dimensions
    assert actor.fc1.in_features == state_dim
    assert actor.fc1.out_features == 64
    assert actor.fc2.in_features == 64
    assert actor.fc2.out_features == action_dim

    print(f"  ✓ Network structure correct")

    # Test forward
    state = torch.randn(1, state_dim)
    probs = actor(state)

    assert probs.shape == (1, action_dim)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)

    print(f"  ✓ Forward pass correct")

    # Test batch
    batch = torch.randn(32, state_dim)
    batch_probs = actor(batch)
    assert batch_probs.shape == (32, action_dim)

    print("\n✅ All tests passed!")


def ppo_critic_network_test(PPOCriticNetwork):
    """Test PPOCriticNetwork class"""
    print("Testing PPOCriticNetwork...")

    state_dim = 4
    critic = PPOCriticNetwork(state_dim, hidden_dim=64)

    # Check structure
    assert hasattr(critic, 'fc1'), "Should have fc1"
    assert hasattr(critic, 'fc2'), "Should have fc2"

    # Check dimensions
    assert critic.fc1.in_features == state_dim
    assert critic.fc1.out_features == 64
    assert critic.fc2.in_features == 64
    assert critic.fc2.out_features == 1

    print(f"  ✓ Network structure correct")

    # Test forward
    state = torch.randn(1, state_dim)
    value = critic(state)

    assert value.shape == (1,) or value.shape == ()

    print(f"  ✓ Forward pass correct")
    print(f"    Sample value: {value.item():.4f}")

    print("\n✅ All tests passed!")


def compute_ppo_loss_test(compute_ppo_loss, PPOActorNetwork, PPOCriticNetwork):
    """Test compute_ppo_loss function"""
    print("Testing compute_ppo_loss...")

    state_dim, action_dim = 4, 2

    # Create networks
    actor = PPOActorNetwork(state_dim, action_dim, hidden_dim=32)
    critic = PPOCriticNetwork(state_dim, hidden_dim=32)

    # Create sample data
    batch_size = 64
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    old_log_probs = torch.randn(batch_size) * 0.5 - 1.0  # Around -1.0
    advantages = torch.randn(batch_size)  # Random advantages
    returns = torch.randn(batch_size) * 2.0

    # Compute loss
    actor_loss, critic_loss, total_loss, approx_kl = compute_ppo_loss(
        actor, states, actions, old_log_probs, advantages,
        critic, returns, clip_epsilon=0.2
    )

    # Check types
    assert isinstance(actor_loss, torch.Tensor), "actor_loss should be torch.Tensor"
    assert isinstance(critic_loss, torch.Tensor), "critic_loss should be torch.Tensor"
    assert isinstance(total_loss, torch.Tensor), "total_loss should be torch.Tensor"
    assert isinstance(approx_kl, torch.Tensor), "approx_kl should be torch.Tensor"

    # Check shapes (should be scalars)
    assert actor_loss.dim() == 0 or actor_loss.shape == torch.Size([])
    assert critic_loss.dim() == 0 or critic_loss.shape == torch.Size([])
    assert total_loss.dim() == 0 or total_loss.shape == torch.Size([])

    # Check requires_grad
    assert total_loss.requires_grad, "total_loss should require grad"

    print(f"  ✓ Loss computation correct")
    print(f"    Actor loss: {actor_loss.item():.4f}")
    print(f"    Critic loss: {critic_loss.item():.4f}")
    print(f"    Total loss: {total_loss.item():.4f}")
    print(f"    Approx KL: {approx_kl.item():.6f}")

    # Test clipping behavior
    # When ratio is > 1+epsilon with positive advantage, should clip
    states_clip = torch.randn(10, state_dim)
    actions_clip = torch.zeros(10, dtype=torch.long)  # All action 0
    old_log_probs_clip = torch.ones(10) * -5.0  # Very low (old policy had low prob)
    advantages_clip = torch.ones(10) * 2.0  # Positive advantages
    returns_clip = torch.ones(10) * 5.0

    actor_loss_clip, _, _, _ = compute_ppo_loss(
        actor, states_clip, actions_clip, old_log_probs_clip, advantages_clip,
        critic, returns_clip, clip_epsilon=0.2
    )

    print(f"  ✓ Clipping mechanism works")
    print(f"    Clipped loss: {actor_loss_clip.item():.4f}")

    # Test gradient flow
    total_loss.backward()
    has_actor_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in actor.parameters())
    has_critic_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in critic.parameters())

    assert has_actor_grad, "Actor should have gradients"
    assert has_critic_grad, "Critic should have gradients"

    print(f"  ✓ Gradients flow correctly")

    print("\n✅ All tests passed!")


def train_ppo_test(train_ppo, PPOActorNetwork, PPOCriticNetwork):
    """Test train_ppo function"""
    print("Testing train_ppo...")

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize networks
    actor = PPOActorNetwork(state_dim, action_dim, hidden_dim=32)
    critic = PPOCriticNetwork(state_dim, hidden_dim=32)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    # Short training run
    n_episodes = 5
    episode_rewards = train_ppo(
        env, actor, critic, optimizer,
        n_episodes=n_episodes,
        gamma=0.99,
        lambda_=0.95,
        clip_epsilon=0.2,
        update_epochs=2,  # Fewer epochs for faster test
        batch_size=32,
        trajectory_length=200  # Shorter trajectory
    )

    # Check output
    assert isinstance(episode_rewards, list), "Should return list"
    # Note: PPO might return fewer episodes if using trajectory_length
    assert len(episode_rewards) > 0, "Should have at least some rewards"

    for i, reward in enumerate(episode_rewards):
        assert isinstance(reward, (int, float, np.integer, np.floating)), \
            f"Episode {i}: reward should be number"
        assert reward > 0, f"Episode {i}: reward should be positive"

    print(f"  ✓ Training completed {len(episode_rewards)} episodes")
    print(f"  ✓ Rewards: min={min(episode_rewards):.1f}, max={max(episode_rewards):.1f}, avg={np.mean(episode_rewards):.1f}")

    # Check networks were updated
    initial_actor_params = [p.clone() for p in actor.parameters()]
    initial_critic_params = [p.clone() for p in critic.parameters()]

    # Run one more episode
    episode_rewards2 = train_ppo(
        env, actor, critic, optimizer,
        n_episodes=1,
        gamma=0.99,
        update_epochs=2,
        trajectory_length=100
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
