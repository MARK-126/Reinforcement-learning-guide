"""
Utility functions for Policy Gradient Tutorial
Functions for testing and supporting Policy Gradient implementations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List
import pytest


def test_implement_policy_network():
    """Test cases for Exercise 1: implement_policy_network"""

    def test_shape_discrete():
        """Test policy network output shape for discrete actions"""
        batch_size, state_dim, action_dim = 4, 8, 3
        hidden_dims = [64, 64]

        # Create network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        network = nn.Sequential(*layers)

        # Test forward pass
        states = torch.randn(batch_size, state_dim)
        logits = network(states)

        assert logits.shape == (batch_size, action_dim), \
            f"Expected shape {(batch_size, action_dim)}, got {logits.shape}"
        print("✓ Discrete action network shape test passed")

    def test_shape_continuous():
        """Test policy network output for continuous actions"""
        batch_size, state_dim, action_dim = 4, 8, 2
        hidden_dims = [64, 64]

        # Create network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        mean_layer = nn.Linear(prev_dim, action_dim)
        log_std_layer = nn.Linear(prev_dim, action_dim)

        # Test forward pass
        states = torch.randn(batch_size, state_dim)
        shared_output = nn.Sequential(*layers)(states)
        mean = mean_layer(shared_output)
        log_std = log_std_layer(shared_output)

        assert mean.shape == (batch_size, action_dim), \
            f"Expected mean shape {(batch_size, action_dim)}, got {mean.shape}"
        assert log_std.shape == (batch_size, action_dim), \
            f"Expected log_std shape {(batch_size, action_dim)}, got {log_std.shape}"
        print("✓ Continuous action network shape test passed")

    test_shape_discrete()
    test_shape_continuous()
    print("All policy network tests passed!")


def test_implement_reinforce_loss():
    """Test cases for Exercise 2: implement_reinforce_loss"""

    def test_loss_shape():
        """Test REINFORCE loss shape and sign"""
        batch_size = 10
        log_probs = torch.randn(batch_size)
        returns = torch.randn(batch_size) * 10 + 100  # Random positive returns

        # REINFORCE loss: -log_prob * return
        loss = -(log_probs * returns).mean()

        assert loss.shape == (), \
            f"Expected scalar loss, got shape {loss.shape}"
        assert loss.item() > 0, \
            "Loss should be positive (we're maximizing returns)"
        print("✓ REINFORCE loss shape and sign test passed")

    def test_loss_value():
        """Test REINFORCE loss numerical value"""
        log_probs = torch.tensor([np.log(0.5), np.log(0.3)])
        returns = torch.tensor([100.0, 50.0])

        # Expected: -mean(log_prob * return)
        expected_loss = -(log_probs * returns).mean()

        loss = -(log_probs * returns).mean()
        assert np.allclose(loss.item(), expected_loss.item()), \
            "Loss value doesn't match expected calculation"
        print("✓ REINFORCE loss value test passed")

    test_loss_shape()
    test_loss_value()
    print("All REINFORCE loss tests passed!")


def test_implement_baseline():
    """Test cases for Exercise 3: implement_baseline"""

    def test_baseline_network_shape():
        """Test baseline network output shape"""
        batch_size, state_dim = 8, 10
        hidden_dims = [64, 64]

        # Create value network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        network = nn.Sequential(*layers)

        # Test forward pass
        states = torch.randn(batch_size, state_dim)
        values = network(states).squeeze(-1)

        assert values.shape == (batch_size,), \
            f"Expected shape {(batch_size,)}, got {values.shape}"
        print("✓ Baseline network shape test passed")

    def test_advantage_computation():
        """Test advantage = return - baseline"""
        returns = torch.tensor([100.0, 150.0, 80.0])
        values = torch.tensor([90.0, 140.0, 85.0])

        advantages = returns - values
        expected = torch.tensor([10.0, 10.0, -5.0])

        assert torch.allclose(advantages, expected), \
            "Advantage computation doesn't match expected values"
        print("✓ Advantage computation test passed")

    test_baseline_network_shape()
    test_advantage_computation()
    print("All baseline tests passed!")


def test_implement_actor_critic():
    """Test cases for Exercise 4: implement_actor_critic"""

    def test_actor_critic_loss_components():
        """Test actor and critic loss components"""
        batch_size = 8

        # Actor loss: -log_prob * advantage
        log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss: MSE(values, returns)
        values = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        critic_loss = ((values - returns) ** 2).mean()

        assert actor_loss.shape == (), "Actor loss should be scalar"
        assert critic_loss.shape == (), "Critic loss should be scalar"
        print("✓ Actor-Critic loss components test passed")

    def test_td_error():
        """Test temporal difference error"""
        reward = 1.0
        gamma = 0.99
        value_current = torch.tensor(50.0)
        value_next = torch.tensor(55.0)
        done = False

        # TD error: r + γV(s') - V(s)
        td_error = reward + gamma * value_next * (1 - done) - value_current

        assert td_error.shape == (), "TD error should be scalar"
        print("✓ TD error computation test passed")

    test_actor_critic_loss_components()
    test_td_error()
    print("All actor-critic tests passed!")


def test_implement_gae():
    """Test cases for Exercise 5: implement_gae"""

    def test_gae_computation():
        """Test GAE computation"""
        rewards = [1.0, 1.0, 1.0]
        values = torch.tensor([10.0, 9.0, 8.0])
        next_value = torch.tensor(0.0)
        gamma = 0.99
        gae_lambda = 0.95
        dones = [False, False, False]

        # Compute GAE backwards
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]

            delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        assert len(advantages) == len(rewards), \
            "GAE should have same length as returns"
        print("✓ GAE computation test passed")

    def test_gae_boundary_cases():
        """Test GAE with early termination"""
        rewards = [1.0, 1.0, 1.0]
        values = torch.tensor([10.0, 9.0, 8.0])
        next_value = torch.tensor(0.0)
        gamma = 0.99
        gae_lambda = 0.95
        dones = [False, False, True]  # Episode ends at t=2

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value if not dones[t] else torch.tensor(0.0)
            else:
                next_v = values[t + 1] * (1 - dones[t])

            delta = rewards[t] + gamma * next_v - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        # At terminal state, GAE should reset
        assert len(advantages) == 3, "Should compute GAE for all timesteps"
        print("✓ GAE boundary cases test passed")

    test_gae_computation()
    test_gae_boundary_cases()
    print("All GAE tests passed!")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("RUNNING ALL POLICY GRADIENT TESTS")
    print("="*70 + "\n")

    print("Testing Exercise 1: Policy Network")
    print("-" * 70)
    test_implement_policy_network()
    print()

    print("Testing Exercise 2: REINFORCE Loss")
    print("-" * 70)
    test_implement_reinforce_loss()
    print()

    print("Testing Exercise 3: Baseline (Value Network)")
    print("-" * 70)
    test_implement_baseline()
    print()

    print("Testing Exercise 4: Actor-Critic")
    print("-" * 70)
    test_implement_actor_critic()
    print()

    print("Testing Exercise 5: GAE")
    print("-" * 70)
    test_implement_gae()
    print()

    print("="*70)
    print("ALL TESTS PASSED!")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
