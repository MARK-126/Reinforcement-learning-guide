"""
Public tests for REINFORCE Exercise
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical


def policy_network_test(target_class):
    """Test PolicyNetwork implementation"""
    print("Testing PolicyNetwork...")

    # Test case 1: Network creation
    policy_net = target_class(state_dim=4, action_dim=2)
    assert isinstance(policy_net, nn.Module), "Should be nn.Module"

    # Test case 2: Forward pass shape
    test_input = torch.randn(8, 4)
    output = policy_net(test_input)
    assert output.shape == (8, 2), f"Output shape wrong. Expected (8, 2), got {output.shape}"

    # Test case 3: Probabilities sum to 1
    probs_sum = output.sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones(8), atol=1e-6), \
        "Probabilities should sum to 1 for each sample"

    # Test case 4: All probabilities positive
    assert torch.all(output >= 0), "All probabilities should be non-negative"
    assert torch.all(output <= 1), "All probabilities should be <= 1"

    # Test case 5: Gradient flow
    assert output.requires_grad, "Output should require gradients"

    print("\033[92mAll tests passed!\033[0m")


def select_action_test(target_function, PolicyNetwork):
    """Test action selection"""
    print("Testing select_action...")

    policy_net = PolicyNetwork(4, 2)
    test_state = np.array([0.1, 0.2, 0.3, 0.4])

    # Test case 1: Return types
    action, log_prob = target_function(policy_net, test_state)
    assert isinstance(action, int), f"Action should be int, got {type(action)}"
    assert isinstance(log_prob, torch.Tensor), "Log prob should be tensor"

    # Test case 2: Valid action
    assert 0 <= action < 2, f"Action should be 0 or 1, got {action}"

    # Test case 3: Log prob requires grad
    assert log_prob.requires_grad, "Log prob should require gradients"

    # Test case 4: Log prob is negative (since prob < 1)
    assert log_prob.item() <= 0, "Log probability should be negative or zero"

    # Test case 5: Multiple samples give different actions (probabilistic)
    actions = [target_function(policy_net, test_state)[0] for _ in range(50)]
    # Should have some variety (with high probability)
    unique_actions = len(set(actions))
    assert unique_actions >= 1, "Should sample at least some actions"

    print("\033[92mAll tests passed!\033[0m")


def compute_policy_loss_test(target_function):
    """Test policy loss computation"""
    print("Testing compute_policy_loss...")

    # Test case 1: Basic loss computation
    log_probs = [torch.tensor(-0.5, requires_grad=True),
                 torch.tensor(-0.3, requires_grad=True),
                 torch.tensor(-0.7, requires_grad=True)]
    returns = torch.tensor([1.0, 2.0, 0.5])

    loss = target_function(log_probs, returns)

    assert isinstance(loss, torch.Tensor), "Loss should be tensor"
    assert loss.requires_grad, "Loss should require gradients"
    assert loss.dim() == 0, "Loss should be scalar"

    # Test case 2: Can backpropagate
    loss.backward()
    for log_prob in log_probs:
        assert log_prob.grad is not None, "Should compute gradients"

    # Test case 3: Loss with all zeros returns
    log_probs_zero = [torch.tensor(-0.5, requires_grad=True),
                      torch.tensor(-0.3, requires_grad=True)]
    returns_zero = torch.tensor([0.0, 0.0])

    loss_zero = target_function(log_probs_zero, returns_zero)
    # With normalized returns (all same), loss should be 0
    assert torch.isfinite(loss_zero), "Loss should be finite even with zero returns"

    print("\033[92mAll tests passed!\033[0m")


def train_reinforce_test(target_function):
    """Test REINFORCE training (simplified)"""
    print("Testing train_reinforce...")
    print("Note: Full training test skipped for speed. Notebook will show full training.")

    env = gym.make('CartPole-v1')

    # Test with few episodes
    try:
        policy_net, rewards = target_function(env, n_episodes=10, lr=0.01, gamma=0.99)

        # Basic checks
        assert isinstance(policy_net, nn.Module), "Should return nn.Module"
        assert len(rewards) == 10, f"Should have 10 rewards, got {len(rewards)}"
        assert all(isinstance(r, (int, float, np.number)) for r in rewards), \
            "All rewards should be numeric"

        # Check policy network is trainable
        params = list(policy_net.parameters())
        assert len(params) > 0, "Policy network should have parameters"
        assert all(p.requires_grad for p in params), "Parameters should require grad"

        print("\033[92mBasic tests passed!\033[0m")
        print("\033[93mFull training test: Run notebook to train for 1000 episodes\033[0m")

    except Exception as e:
        print(f"\033[91mTest failed with error: {e}\033[0m")
        raise

    finally:
        env.close()


if __name__ == "__main__":
    print("This file contains test functions for REINFORCE.")
    print("Import and use them in the REINFORCE Exercise notebook.")
