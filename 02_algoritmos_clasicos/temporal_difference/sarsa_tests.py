"""
Public tests for SARSA Exercise

This file contains test functions to validate student implementations
of the SARSA algorithm.
"""

import numpy as np
import gymnasium as gym


def sarsa_update_test(target):
    """Test SARSA update rule"""
    print("Testing sarsa_update...")

    # Test case 1: Non-terminal update with specific next_action
    Q = np.zeros((4, 2))
    Q[1] = [0.5, 0.3]  # Q-values for next_state

    alpha = 0.1
    gamma = 0.9

    # SARSA uses next_action=1 (Q[1,1]=0.3), NOT max (Q[1,0]=0.5)
    Q_updated, td_error = target(
        Q.copy(), state=0, action=0, reward=1.0,
        next_state=1, next_action=1, done=False,
        alpha=alpha, gamma=gamma
    )

    # Expected: Q[0,0] = 0 + 0.1 * (1.0 + 0.9 * 0.3 - 0) = 0.127
    expected_q = 0.127
    expected_td = 1.27

    assert np.isclose(Q_updated[0, 0], expected_q, atol=1e-6), \
        f"SARSA update failed. Expected Q[0,0]={expected_q:.4f}, got {Q_updated[0, 0]:.4f}"
    assert np.isclose(td_error, expected_td, atol=1e-6), \
        f"TD error wrong. Expected {expected_td:.4f}, got {td_error:.4f}"

    # Test case 2: Verify SARSA uses next_action, not max
    # If next_action=0 (max Q-value), should get different result
    Q_updated2, td_error2 = target(
        Q.copy(), state=0, action=0, reward=1.0,
        next_state=1, next_action=0, done=False,  # next_action=0 this time
        alpha=alpha, gamma=gamma
    )

    # Expected: Q[0,0] = 0 + 0.1 * (1.0 + 0.9 * 0.5 - 0) = 0.145
    expected_q2 = 0.145
    expected_td2 = 1.45

    assert np.isclose(Q_updated2[0, 0], expected_q2, atol=1e-6), \
        f"SARSA should use next_action. Expected Q[0,0]={expected_q2:.4f}, got {Q_updated2[0, 0]:.4f}"

    # Verify different next_actions give different results
    assert not np.isclose(Q_updated[0, 0], Q_updated2[0, 0]), \
        "SARSA should give different results for different next_actions!"

    # Test case 3: Terminal update
    Q = np.zeros((4, 2))
    Q[1] = [0.5, 0.3]

    Q_updated, td_error = target(
        Q.copy(), state=0, action=0, reward=1.0,
        next_state=1, next_action=0, done=True,
        alpha=alpha, gamma=gamma
    )

    # Expected: Q[0,0] = 0 + 0.1 * (1.0 - 0) = 0.1
    expected_q = 0.1
    expected_td = 1.0

    assert np.isclose(Q_updated[0, 0], expected_q, atol=1e-6), \
        f"Terminal SARSA update failed. Expected Q[0,0]={expected_q:.4f}, got {Q_updated[0, 0]:.4f}"
    assert np.isclose(td_error, expected_td, atol=1e-6), \
        f"TD error wrong for terminal. Expected {expected_td:.4f}, got {td_error:.4f}"

    # Test case 4: Update with existing Q-value
    Q = np.array([[0.2, 0.3], [0.4, 0.6], [0.1, 0.5], [0.8, 0.2]])

    Q_updated, td_error = target(
        Q.copy(), state=0, action=1, reward=0.5,
        next_state=2, next_action=0, done=False,
        alpha=0.2, gamma=0.95
    )

    # Current Q[0,1] = 0.3
    # Next action Q[2,0] = 0.1 (NOT max which is 0.5)
    # Target = 0.5 + 0.95 * 0.1 = 0.595
    # TD error = 0.595 - 0.3 = 0.295
    # New Q[0,1] = 0.3 + 0.2 * 0.295 = 0.359
    expected_q = 0.359
    expected_td = 0.295

    assert np.isclose(Q_updated[0, 1], expected_q, atol=1e-6), \
        f"SARSA with existing Q failed. Expected Q[0,1]={expected_q:.4f}, got {Q_updated[0, 1]:.4f}"
    assert np.isclose(td_error, expected_td, atol=1e-6), \
        f"TD error wrong. Expected {expected_td:.4f}, got {td_error:.4f}"

    print("\033[92mAll tests passed!\033[0m")


def train_sarsa_test(target):
    """Test SARSA training loop"""
    print("Testing train_sarsa...")

    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=False)

    # Test case 1: Basic training
    np.random.seed(42)
    Q, rewards_history = target(
        env,
        n_episodes=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        max_steps=50
    )

    # Check Q-table shape
    assert Q.shape == (16, 4), f"Q-table shape wrong. Expected (16, 4), got {Q.shape}"

    # Check rewards history length
    assert len(rewards_history) == 100, \
        f"Rewards history length wrong. Expected 100, got {len(rewards_history)}"

    # Check that Q-values have been updated
    assert not np.allclose(Q, 0), "Q-table should be updated (not all zeros)"

    # Test case 2: Agent should learn over time
    np.random.seed(42)
    Q, rewards_history = target(
        env,
        n_episodes=500,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        max_steps=100
    )

    early_avg = np.mean(rewards_history[:100])
    late_avg = np.mean(rewards_history[-100:])

    assert late_avg >= early_avg, \
        f"SARSA should improve over time. Early: {early_avg:.3f}, Late: {late_avg:.3f}"

    # Test case 3: Should eventually solve FrozenLake
    assert late_avg > 0.5, \
        f"SARSA should learn to solve FrozenLake. Got success rate: {late_avg:.3f}"

    env.close()

    print("\033[92mAll tests passed!\033[0m")
    print("\n\033[93mCongratulations! Your SARSA implementation is correct!\033[0m")


if __name__ == "__main__":
    print("This file contains test functions for SARSA.")
    print("Import and use them in the SARSA Exercise notebook.")
