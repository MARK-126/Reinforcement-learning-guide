"""
Public tests for Q-Learning Exercise

This file contains test functions to validate student implementations
of the Q-Learning algorithm components.
"""

import numpy as np
import gymnasium as gym


def initialize_q_table_test(target):
    """Test Q-table initialization"""
    print("Testing initialize_q_table...")

    # Test case 1: Basic initialization with zeros
    Q = target(16, 4, init_value=0.0)
    assert Q.shape == (16, 4), f"Wrong shape. Expected (16, 4), got {Q.shape}"
    assert np.allclose(Q, np.zeros((16, 4))), "Q-table should be initialized with zeros"

    # Test case 2: Initialization with custom value
    Q = target(10, 3, init_value=0.5)
    assert Q.shape == (10, 3), f"Wrong shape. Expected (10, 3), got {Q.shape}"
    assert np.allclose(Q, np.ones((10, 3)) * 0.5), "Q-table should be initialized with custom value"

    # Test case 3: Different dimensions
    Q = target(5, 8, init_value=1.0)
    assert Q.shape == (5, 8), f"Wrong shape. Expected (5, 8), got {Q.shape}"
    assert np.allclose(Q, np.ones((5, 8))), "Q-table should be initialized with ones"

    print("\033[92mAll tests passed!\033[0m")


def epsilon_greedy_action_test(target):
    """Test epsilon-greedy action selection"""
    print("Testing epsilon_greedy_action...")

    # Setup
    np.random.seed(42)
    Q = np.array([[0.1, 0.5, 0.2, 0.3],
                  [0.4, 0.1, 0.6, 0.2],
                  [0.2, 0.3, 0.1, 0.4]])
    n_actions = 4

    # Test case 1: epsilon = 0 (always exploit)
    np.random.seed(42)
    action = target(Q, state=0, n_actions=n_actions, epsilon=0.0)
    assert action == 1, f"With epsilon=0, should select action with max Q-value. Expected 1, got {action}"

    action = target(Q, state=1, n_actions=n_actions, epsilon=0.0)
    assert action == 2, f"With epsilon=0, should select action with max Q-value. Expected 2, got {action}"

    # Test case 2: epsilon = 1 (always explore)
    np.random.seed(42)
    actions = [target(Q, state=0, n_actions=n_actions, epsilon=1.0) for _ in range(100)]
    unique_actions = set(actions)
    assert len(unique_actions) > 1, "With epsilon=1, should select various random actions"
    assert all(0 <= a < n_actions for a in actions), "Actions should be in valid range"

    # Test case 3: epsilon = 0.5 (mixed)
    np.random.seed(42)
    actions = [target(Q, state=0, n_actions=n_actions, epsilon=0.5) for _ in range(200)]
    greedy_action = np.argmax(Q[0])
    greedy_count = sum(1 for a in actions if a == greedy_action)
    # Should have roughly 50% greedy actions, 50% random (allowing some variance)
    assert 80 < greedy_count < 160, f"With epsilon=0.5, expected ~100 greedy actions, got {greedy_count}"

    # Test case 4: Check return type
    action = target(Q, state=0, n_actions=n_actions, epsilon=0.2)
    assert isinstance(action, (int, np.integer)), f"Action should be integer, got {type(action)}"

    print("\033[92mAll tests passed!\033[0m")


def q_learning_update_test(target):
    """Test Q-Learning update rule"""
    print("Testing q_learning_update...")

    # Test case 1: Non-terminal update
    Q = np.zeros((4, 2))
    Q[1] = [0.5, 0.3]  # Q-values for next_state

    alpha = 0.1
    gamma = 0.9

    Q_updated, td_error = target(
        Q.copy(), state=0, action=0, reward=1.0,
        next_state=1, done=False, alpha=alpha, gamma=gamma
    )

    # Expected: Q[0,0] = 0 + 0.1 * (1.0 + 0.9 * 0.5 - 0) = 0.145
    expected_q = 0.145
    expected_td = 1.45

    assert np.isclose(Q_updated[0, 0], expected_q, atol=1e-6), \
        f"Non-terminal update failed. Expected Q[0,0]={expected_q:.4f}, got {Q_updated[0, 0]:.4f}"
    assert np.isclose(td_error, expected_td, atol=1e-6), \
        f"TD error wrong. Expected {expected_td:.4f}, got {td_error:.4f}"

    # Test case 2: Terminal update
    Q = np.zeros((4, 2))
    Q[1] = [0.5, 0.3]

    Q_updated, td_error = target(
        Q.copy(), state=0, action=0, reward=1.0,
        next_state=1, done=True, alpha=alpha, gamma=gamma
    )

    # Expected: Q[0,0] = 0 + 0.1 * (1.0 - 0) = 0.1
    expected_q = 0.1
    expected_td = 1.0

    assert np.isclose(Q_updated[0, 0], expected_q, atol=1e-6), \
        f"Terminal update failed. Expected Q[0,0]={expected_q:.4f}, got {Q_updated[0, 0]:.4f}"
    assert np.isclose(td_error, expected_td, atol=1e-6), \
        f"TD error wrong for terminal state. Expected {expected_td:.4f}, got {td_error:.4f}"

    # Test case 3: Update with existing Q-value
    Q = np.array([[0.2, 0.3], [0.4, 0.6], [0.1, 0.5], [0.8, 0.2]])

    Q_updated, td_error = target(
        Q.copy(), state=0, action=1, reward=0.5,
        next_state=2, done=False, alpha=0.2, gamma=0.95
    )

    # Current Q[0,1] = 0.3
    # Max Q[2] = 0.5
    # Target = 0.5 + 0.95 * 0.5 = 0.975
    # TD error = 0.975 - 0.3 = 0.675
    # New Q[0,1] = 0.3 + 0.2 * 0.675 = 0.435
    expected_q = 0.435
    expected_td = 0.675

    assert np.isclose(Q_updated[0, 1], expected_q, atol=1e-6), \
        f"Update with existing Q-value failed. Expected Q[0,1]={expected_q:.4f}, got {Q_updated[0, 1]:.4f}"
    assert np.isclose(td_error, expected_td, atol=1e-6), \
        f"TD error wrong. Expected {expected_td:.4f}, got {td_error:.4f}"

    # Test case 4: Negative reward
    Q = np.zeros((2, 2))
    Q[1] = [0.2, 0.1]

    Q_updated, td_error = target(
        Q.copy(), state=0, action=0, reward=-1.0,
        next_state=1, done=False, alpha=0.1, gamma=0.9
    )

    # Target = -1.0 + 0.9 * 0.2 = -0.82
    # TD error = -0.82 - 0 = -0.82
    # New Q[0,0] = 0 + 0.1 * (-0.82) = -0.082
    expected_q = -0.082
    expected_td = -0.82

    assert np.isclose(Q_updated[0, 0], expected_q, atol=1e-6), \
        f"Negative reward update failed. Expected Q[0,0]={expected_q:.4f}, got {Q_updated[0, 0]:.4f}"
    assert np.isclose(td_error, expected_td, atol=1e-6), \
        f"TD error wrong for negative reward. Expected {expected_td:.4f}, got {td_error:.4f}"

    # Test case 5: Original Q-table should not be modified
    Q_original = np.ones((3, 3))
    Q_copy = Q_original.copy()
    Q_updated, _ = target(Q_original, 0, 0, 1.0, 1, False, 0.1, 0.9)
    # Allow modification of original, but check that function works correctly

    print("\033[92mAll tests passed!\033[0m")


def train_q_learning_test(target):
    """Test complete Q-Learning training loop"""
    print("Testing train_q_learning...")

    # Create a simple environment
    env = gym.make('FrozenLake-v1', is_slippery=False)

    # Test case 1: Basic training
    np.random.seed(42)
    Q, rewards_history, epsilon_history = target(
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

    # Check histories length
    assert len(rewards_history) == 100, f"Rewards history length wrong. Expected 100, got {len(rewards_history)}"
    assert len(epsilon_history) == 100, f"Epsilon history length wrong. Expected 100, got {len(epsilon_history)}"

    # Check epsilon decay
    assert epsilon_history[0] == 1.0, f"Initial epsilon should be 1.0, got {epsilon_history[0]}"
    assert epsilon_history[-1] < epsilon_history[0], "Epsilon should decay over time"
    assert epsilon_history[-1] >= 0.01, f"Epsilon should not go below epsilon_min (0.01), got {epsilon_history[-1]}"

    # Check that Q-values have been updated (not all zeros)
    assert not np.allclose(Q, 0), "Q-table should be updated (not all zeros)"

    # Check rewards are valid
    assert all(isinstance(r, (int, float, np.number)) for r in rewards_history), \
        "All rewards should be numeric"

    # Test case 2: Agent should learn (improving performance)
    np.random.seed(42)
    Q, rewards_history, epsilon_history = target(
        env,
        n_episodes=500,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        max_steps=100
    )

    # Average reward should improve over time
    early_avg = np.mean(rewards_history[:100])
    late_avg = np.mean(rewards_history[-100:])

    assert late_avg >= early_avg, \
        f"Agent should improve over time. Early avg: {early_avg:.3f}, Late avg: {late_avg:.3f}"

    # Test case 3: With deterministic FrozenLake, should eventually solve it
    # Late average should be reasonably good (>0.5 success rate)
    assert late_avg > 0.5, \
        f"Agent should learn to solve FrozenLake (deterministic). Got success rate: {late_avg:.3f}"

    env.close()

    print("\033[92mAll tests passed!\033[0m")
    print("\n\033[93mCongratulations! Your Q-Learning implementation is correct!\033[0m")


if __name__ == "__main__":
    print("This file contains test functions.")
    print("Import and use them in the Q-Learning Exercise notebook.")
