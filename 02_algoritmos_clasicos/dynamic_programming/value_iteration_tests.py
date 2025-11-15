"""
Public tests for Value Iteration Exercise

This file contains test functions to validate student implementations
of the Value Iteration algorithm components.
"""

import numpy as np
import gymnasium as gym


def initialize_value_function_test(target):
    """Test value function initialization"""
    print("Testing initialize_value_function...")

    # Test case 1: Basic initialization with zeros
    V = target(n_states=10, terminal_states=None, init_value=0.0)
    assert V.shape == (10,), f"Wrong shape. Expected (10,), got {V.shape}"
    assert np.allclose(V, np.zeros(10)), "V should be all zeros"

    # Test case 2: With terminal states
    V = target(n_states=16, terminal_states=[5, 7, 15], init_value=0.0)
    assert V[5] == 0 and V[7] == 0 and V[15] == 0, "Terminal states should be 0"

    # Test case 3: Non-zero initialization
    V = target(n_states=8, terminal_states=[7], init_value=1.0)
    assert np.allclose(V[:7], 1.0), "Non-terminal states should be init_value"
    assert V[7] == 0, "Terminal state should be 0"

    # Test case 4: Multiple terminal states
    V = target(n_states=16, terminal_states=[5,7,11,12,15], init_value=0.5)
    for term in [5,7,11,12,15]:
        assert V[term] == 0, f"Terminal state {term} should be 0, got {V[term]}"

    print("\033[92mAll tests passed!\033[0m")


def bellman_optimality_backup_test(target):
    """Test Bellman optimality backup"""
    print("Testing bellman_optimality_backup...")

    env = gym.make('FrozenLake-v1', is_slippery=False)

    # Test case 1: State next to goal
    V = np.zeros(16)
    V[15] = 1.0  # Goal state
    new_value, best_action = target(env, V, state=14, gamma=0.99)

    # State 14, action 2 (right) goes to goal with reward 1
    # Expected: 1 + 0.99 * 1 = 1.99, but actually reward 0, so 0 + 0.99 * 1 = 0.99
    assert np.isclose(new_value, 0.99, atol=0.01), \
        f"State 14 value wrong. Expected ~0.99, got {new_value}"

    # Test case 2: Initial state (far from goal)
    V = np.zeros(16)
    V[15] = 1.0
    new_value, best_action = target(env, V, state=0, gamma=0.99)
    assert new_value >= 0, "Value should be non-negative"

    # Test case 3: Check that function returns action
    assert isinstance(best_action, (int, np.integer)), \
        f"best_action should be integer, got {type(best_action)}"
    assert 0 <= best_action < 4, f"Action should be 0-3, got {best_action}"

    # Test case 4: All states with random V
    np.random.seed(42)
    V = np.random.rand(16)
    for state in range(16):
        new_value, best_action = target(env, V, state=state, gamma=0.99)
        assert not np.isnan(new_value), f"NaN value for state {state}"
        assert 0 <= best_action < 4, f"Invalid action {best_action} for state {state}"

    env.close()
    print("\033[92mAll tests passed!\033[0m")


def extract_policy_test(target):
    """Test policy extraction"""
    print("Testing extract_policy...")

    env = gym.make('FrozenLake-v1', is_slippery=False)

    # Test case 1: Policy shape
    V = np.zeros(16)
    policy = target(env, V, gamma=0.99)
    assert policy.shape == (16,), f"Policy shape wrong. Expected (16,), got {policy.shape}"

    # Test case 2: Policy values are valid actions
    assert np.all((policy >= 0) & (policy < 4)), "Policy contains invalid actions"

    # Test case 3: With goal-biased value function
    V = np.zeros(16)
    V[15] = 1.0
    policy = target(env, V, gamma=0.99)

    # State 14 should choose right (action 2) to go to goal
    assert policy[14] == 2, f"State 14 should choose right (2), got {policy[14]}"

    # Test case 4: All actions should be integers
    for action in policy:
        assert isinstance(action, (int, np.integer)), \
            f"Policy action should be integer, got {type(action)}"

    env.close()
    print("\033[92mAll tests passed!\033[0m")


def value_iteration_sweep_test(target):
    """Test value iteration sweep"""
    print("Testing value_iteration_sweep...")

    env = gym.make('FrozenLake-v1', is_slippery=False)

    # Test case 1: Single sweep
    V = np.zeros(16)
    V_updated, delta = target(env, V, gamma=0.99)

    assert V_updated.shape == (16,), "Updated V should have same shape"
    assert isinstance(delta, (float, np.floating)), \
        f"Delta should be float, got {type(delta)}"
    assert delta >= 0, "Delta should be non-negative"

    # Test case 2: Values should change (not all zeros after sweep)
    assert not np.allclose(V_updated, V), "V should change after sweep"

    # Test case 3: Delta should decrease over sweeps
    V = np.zeros(16)
    deltas = []
    for _ in range(5):
        V, delta = target(env, V, gamma=0.99)
        deltas.append(delta)

    # Delta should generally decrease (may have small fluctuations)
    assert deltas[-1] < deltas[0], \
        f"Delta should decrease. First: {deltas[0]:.6f}, Last: {deltas[-1]:.6f}"

    # Test case 4: Check in-place update
    V_before = np.zeros(16)
    V_after, _ = target(env, V_before, gamma=0.99)
    # V_after should be the same object (in-place update)
    # But the values should have changed

    env.close()
    print("\033[92mAll tests passed!\033[0m")


def value_iteration_test(target):
    """Test complete value iteration algorithm"""
    print("Testing value_iteration...")

    env = gym.make('FrozenLake-v1', is_slippery=False)

    # Test case 1: Algorithm converges
    V, policy, n_iter, deltas = target(env, gamma=0.99, theta=1e-6, max_iterations=1000)

    assert V.shape == (16,), f"V shape wrong. Expected (16,), got {V.shape}"
    assert policy.shape == (16,), f"Policy shape wrong. Expected (16,), got {policy.shape}"
    assert isinstance(n_iter, int), f"n_iter should be int, got {type(n_iter)}"
    assert n_iter < 1000, f"Should converge before max_iterations. Got {n_iter}"

    # Test case 2: Deltas list
    assert len(deltas) == n_iter, \
        f"Deltas length should match iterations. {len(deltas)} != {n_iter}"
    assert deltas[-1] < 1e-6, \
        f"Final delta should be below threshold. Got {deltas[-1]}"

    # Test case 3: Goal state value
    # Goal state (15) should have value related to reaching it
    assert V[15] >= 0, f"Goal state value should be non-negative. Got {V[15]}"

    # Test case 4: Policy is valid
    assert np.all((policy >= 0) & (policy < 4)), "Policy contains invalid actions"

    # Test case 5: Value function quality
    # States closer to goal should generally have higher values
    # State 14 (next to goal) should have high value
    assert V[14] > V[0], \
        f"State 14 (near goal) should have higher value than state 0. V[14]={V[14]:.3f}, V[0]={V[0]:.3f}"

    # Test case 6: Convergence behavior
    # Deltas should decrease over time
    if len(deltas) > 2:
        assert deltas[-1] < deltas[0], \
            "Deltas should decrease toward convergence"

    env.close()
    print("\033[92mAll tests passed!\033[0m")
    print("\n\033[93mCongratulations! Your Value Iteration implementation is correct!\033[0m")


if __name__ == "__main__":
    print("This file contains test functions for Value Iteration.")
    print("Import and use them in the Value Iteration Exercise notebook.")
