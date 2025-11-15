"""
Public tests for Monte Carlo Control Exercise
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict


def generate_episode_test(target):
    """Test episode generation"""
    print("Testing generate_episode...")

    env = gym.make('FrozenLake-v1', is_slippery=False)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Test case 1: Episode is a list
    episode = target(env, Q, epsilon=0.5)
    assert isinstance(episode, list), "Episode should be a list"

    # Test case 2: Episode contains tuples of (state, action, reward)
    if len(episode) > 0:
        assert len(episode[0]) == 3, "Each step should be (state, action, reward)"
        state, action, reward = episode[0]
        assert isinstance(state, (int, np.integer)), "State should be integer"
        assert isinstance(action, (int, np.integer)), "Action should be integer"
        assert isinstance(reward, (int, float, np.number)), "Reward should be numeric"

    # Test case 3: Actions are valid
    for state, action, reward in episode:
        assert 0 <= action < env.action_space.n, f"Invalid action {action}"

    # Test case 4: Episode length is reasonable
    assert len(episode) > 0, "Episode should have at least one step"
    assert len(episode) <= 100, "Episode too long (check max_steps)"

    env.close()
    print("\033[92mAll tests passed!\033[0m")


def calculate_returns_test(target):
    """Test return calculation"""
    print("Testing calculate_returns...")

    # Test case 1: Simple episode
    episode = [(0, 1, 0), (1, 2, 0), (2, 1, 1)]
    returns = target(episode, gamma=0.9)

    assert len(returns) == len(episode), "Returns length should match episode length"

    # Manual calculation:
    # G[2] = 1
    # G[1] = 0 + 0.9 * 1 = 0.9
    # G[0] = 0 + 0.9 * 0.9 = 0.81
    expected = [0.81, 0.9, 1.0]
    assert np.allclose(returns, expected), \
        f"Returns wrong. Expected {expected}, got {returns}"

    # Test case 2: All zero rewards
    episode_zeros = [(0, 0, 0), (1, 1, 0), (2, 2, 0)]
    returns_zeros = target(episode_zeros, gamma=0.9)
    assert np.allclose(returns_zeros, [0, 0, 0]), "All zeros should give zero returns"

    # Test case 3: Gamma = 1
    episode_simple = [(0, 0, 1), (1, 1, 1)]
    returns_gamma1 = target(episode_simple, gamma=1.0)
    assert np.allclose(returns_gamma1, [2, 1]), "With gamma=1, should be simple sum"

    print("\033[92mAll tests passed!\033[0m")


def update_q_values_test(target):
    """Test Q-value updates"""
    print("Testing update_q_values...")

    # Test case 1: Basic update
    Q = defaultdict(lambda: np.zeros(4))
    episode = [(0, 1, 0), (1, 2, 0), (2, 1, 1)]
    returns = [0.81, 0.9, 1.0]

    Q_updated = target(Q, episode, returns, alpha=0.1)

    # Q[0][1] should be updated: 0 + 0.1 * (0.81 - 0) = 0.081
    expected_q01 = 0.081
    assert np.isclose(Q_updated[0][1], expected_q01, atol=0.001), \
        f"Q[0][1] wrong. Expected {expected_q01:.3f}, got {Q_updated[0][1]:.3f}"

    # Test case 2: First-visit only
    Q2 = defaultdict(lambda: np.zeros(4))
    episode_repeat = [(0, 1, 0), (1, 2, 0), (0, 1, 1)]  # (0,1) appears twice
    returns_repeat = [1.9, 1.0, 1.0]

    Q2_updated = target(Q2, episode_repeat, returns_repeat, alpha=0.1)

    # Should only update with first visit: 0 + 0.1 * (1.9 - 0) = 0.19
    expected = 0.19
    assert np.isclose(Q2_updated[0][1], expected, atol=0.001), \
        "First-visit MC should only update on first occurrence"

    print("\033[92mAll tests passed!\033[0m")


def monte_carlo_control_test(target):
    """Test complete MC Control"""
    print("Testing monte_carlo_control...")

    env = gym.make('FrozenLake-v1', is_slippery=False)

    # Test case 1: Returns correct types
    Q, policy, rewards = target(env, n_episodes=100, alpha=0.1, gamma=0.99)

    assert isinstance(Q, defaultdict), "Q should be defaultdict"
    assert isinstance(policy, (dict, defaultdict, np.ndarray)), "Policy should be dict or array"
    assert isinstance(rewards, list), "Rewards should be list"
    assert len(rewards) == 100, f"Should have 100 rewards, got {len(rewards)}"

    # Test case 2: Learning occurs
    Q, policy, rewards = target(env, n_episodes=1000, alpha=0.1, gamma=0.99)

    early_avg = np.mean(rewards[:100])
    late_avg = np.mean(rewards[-100:])

    assert late_avg >= early_avg, \
        f"Should learn over time. Early: {early_avg:.3f}, Late: {late_avg:.3f}"

    # Test case 3: Eventually solves FrozenLake
    assert late_avg > 0.3, \
        f"Should learn to reach goal sometimes. Got {late_avg:.3f}"

    env.close()
    print("\033[92mAll tests passed!\033[0m")
    print("\n\033[93mCongratulations! Your Monte Carlo Control implementation is correct!\033[0m")


if __name__ == "__main__":
    print("This file contains test functions for Monte Carlo Control.")
    print("Import and use them in the Monte Carlo Exercise notebook.")
