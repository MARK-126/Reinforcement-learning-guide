"""
Utility functions and tests for Dynamic Programming Tutorial

This module provides helper functions and automated tests for the
Dynamic Programming notebook exercises.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import sys
import os

# Add repository path for imports
repo_path = '/home/user/Reinforcement-learning-guide'
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Import Dynamic Programming implementations
import importlib.util

# Load policy_iteration.py
pi_spec = importlib.util.spec_from_file_location(
    "policy_iteration",
    os.path.join(repo_path, "02_algoritmos_clasicos/dynamic_programming/policy_iteration.py")
)
pi_module = importlib.util.module_from_spec(pi_spec)
pi_spec.loader.exec_module(pi_module)

# Load value_iteration.py
vi_spec = importlib.util.spec_from_file_location(
    "value_iteration",
    os.path.join(repo_path, "02_algoritmos_clasicos/dynamic_programming/value_iteration.py")
)
vi_module = importlib.util.module_from_spec(vi_spec)
vi_spec.loader.exec_module(vi_module)

# Import classes and functions
PolicyIteration = pi_module.PolicyIteration
ValueIteration = vi_module.ValueIteration
create_gridworld_mdp = pi_module.create_gridworld_mdp


# ==================== HELPER FUNCTIONS ====================

def visualize_policy_and_values(policy, values, grid_size=4, title=""):
    """
    Visualizes policy and value function in a GridWorld.

    Arguments:
    policy -- numpy array of shape (n_states,) with optimal actions
    values -- numpy array of shape (n_states,) with state values
    grid_size -- integer, size of the grid (default 4)
    title -- string, title suffix for the plots

    Returns:
    None (displays plot)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Action symbols
    action_symbols = ['↑', '→', '↓', '←']

    # Create grids for visualization
    policy_grid = policy.reshape(grid_size, grid_size)
    value_grid = values.reshape(grid_size, grid_size)

    # 1. Visualize Policy
    ax1.set_title(f'Optimal Policy {title}', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, grid_size - 0.5)
    ax1.set_ylim(-0.5, grid_size - 0.5)
    ax1.set_xticks(range(grid_size))
    ax1.set_yticks(range(grid_size))
    ax1.grid(True, linewidth=2)
    ax1.invert_yaxis()

    for i in range(grid_size):
        for j in range(grid_size):
            action = policy_grid[i, j]
            ax1.text(j, i, action_symbols[action],
                    ha='center', va='center', fontsize=24,
                    color='blue' if (i == grid_size-1 and j == grid_size-1) else 'black')

    # Mark start and goal
    ax1.text(0, -0.8, 'START', ha='center', fontsize=10, color='green', fontweight='bold')
    ax1.text(grid_size-1, grid_size-0.2, 'GOAL', ha='center', fontsize=10, color='red', fontweight='bold')

    # 2. Visualize Values
    im = ax2.imshow(value_grid, cmap='RdYlGn', interpolation='nearest')
    ax2.set_title(f'Value Function {title}', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(grid_size))
    ax2.set_yticks(range(grid_size))

    # Add numerical values
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax2.text(j, i, f'{value_grid[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(im, ax=ax2, label='State Value')
    plt.tight_layout()
    plt.show()


def visualize_trajectory(policy, trajectory, grid_size=4, title="Optimal Trajectory"):
    """
    Visualizes a trajectory in the grid.

    Arguments:
    policy -- numpy array of shape (n_states,) with optimal actions
    trajectory -- list of states representing the path
    grid_size -- integer, size of the grid (default 4)
    title -- string, title for the plot

    Returns:
    None (displays plot)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    action_symbols = ['↑', '→', '↓', '←']
    policy_grid = policy.reshape(grid_size, grid_size)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, linewidth=2)
    ax.invert_yaxis()

    # Draw policy
    for i in range(grid_size):
        for j in range(grid_size):
            action = policy_grid[i, j]
            ax.text(j, i, action_symbols[action],
                   ha='center', va='center', fontsize=20, color='lightgray')

    # Draw trajectory
    for idx in range(len(trajectory) - 1):
        s1 = trajectory[idx]
        s2 = trajectory[idx + 1]

        r1, c1 = s1 // grid_size, s1 % grid_size
        r2, c2 = s2 // grid_size, s2 % grid_size

        # Arrow
        arrow = FancyArrowPatch(
            (c1, r1), (c2, r2),
            arrowstyle='->', mutation_scale=30, linewidth=3,
            color='red', alpha=0.7
        )
        ax.add_patch(arrow)

    # Mark start and end
    start = trajectory[0]
    end = trajectory[-1]
    r_start, c_start = start // grid_size, start % grid_size
    r_end, c_end = end // grid_size, end % grid_size

    ax.plot(c_start, r_start, 'go', markersize=20, label='Start')
    ax.plot(c_end, r_end, 'r*', markersize=25, label='Goal')

    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()


def simulate_trajectory(policy, start_state, goal_state, grid_size=4, max_steps=20):
    """
    Simulates a trajectory following the optimal policy.

    Arguments:
    policy -- numpy array of shape (n_states,) with optimal actions
    start_state -- integer, starting state
    goal_state -- integer, goal state
    grid_size -- integer, size of the grid (default 4)
    max_steps -- integer, maximum number of steps (default 20)

    Returns:
    trajectory -- list of states representing the path
    """
    trajectory = [start_state]
    current_state = start_state

    for _ in range(max_steps):
        if current_state == goal_state:
            break

        # Take action according to policy
        action = policy[current_state]

        # Simulate transition (deterministic in our GridWorld)
        row, col = current_state // grid_size, current_state % grid_size

        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(grid_size - 1, col + 1)
        elif action == 2:  # down
            row = min(grid_size - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)

        current_state = row * grid_size + col
        trajectory.append(current_state)

    return trajectory


def create_gridworld_with_penalties(grid_size=4, goal_reward=1.0, step_reward=-0.1):
    """
    Creates a GridWorld MDP with higher step penalties.

    Arguments:
    grid_size -- integer, size of the grid
    goal_reward -- float, reward for reaching the goal
    step_reward -- float, reward for each step (should be negative)

    Returns:
    transition_probs -- numpy array of shape (n_states, n_actions, n_states)
    rewards -- numpy array of shape (n_states, n_actions, n_states)
    n_states -- integer, number of states
    n_actions -- integer, number of actions
    """
    return create_gridworld_mdp(grid_size, goal_reward, step_reward)


# ==================== TEST FUNCTIONS ====================

def policy_evaluation_test(policy_eval_func):
    """
    Test function for policy evaluation implementation.

    Arguments:
    policy_eval_func -- function that implements policy evaluation

    Returns:
    None (prints test results)
    """
    print("Testing policy_evaluation function...")

    # Create simple test case
    trans, rew, n_s, n_a = create_gridworld_mdp(grid_size=3)

    # Initialize uniform random policy
    policy = np.random.randint(0, n_a, size=n_s)

    # Test policy evaluation
    try:
        V = policy_eval_func(policy, trans, rew, gamma=0.9, theta=1e-6)

        # Check shape
        assert V.shape == (n_s,), f"Wrong shape: {V.shape} != {(n_s,)}"

        # Check that values are finite
        assert np.all(np.isfinite(V)), "Values contain NaN or Inf"

        # Check that goal state has highest value (for uniform policy this should be true)
        # This is a weak test but ensures basic sanity
        assert np.max(V) >= 0, "Maximum value should be non-negative"

        print("\033[92m✓ All tests passed!\033[0m")

    except Exception as e:
        print(f"\033[91m✗ Test failed: {str(e)}\033[0m")
        raise


def policy_improvement_test(policy_improve_func):
    """
    Test function for policy improvement implementation.

    Arguments:
    policy_improve_func -- function that implements policy improvement

    Returns:
    None (prints test results)
    """
    print("Testing policy_improvement function...")

    # Create simple test case
    trans, rew, n_s, n_a = create_gridworld_mdp(grid_size=3)

    # Initialize with zeros (suboptimal)
    V = np.zeros(n_s)

    try:
        new_policy = policy_improve_func(V, trans, rew, gamma=0.9)

        # Check shape
        assert new_policy.shape == (n_s,), f"Wrong shape: {new_policy.shape} != {(n_s,)}"

        # Check that policy contains valid actions
        assert np.all((new_policy >= 0) & (new_policy < n_a)), "Policy contains invalid actions"

        # Check that policy is integer type
        assert new_policy.dtype in [np.int32, np.int64, int], "Policy should contain integers"

        print("\033[92m✓ All tests passed!\033[0m")

    except Exception as e:
        print(f"\033[91m✗ Test failed: {str(e)}\033[0m")
        raise


def value_iteration_step_test(value_iter_step_func):
    """
    Test function for value iteration step implementation.

    Arguments:
    value_iter_step_func -- function that implements one step of value iteration

    Returns:
    None (prints test results)
    """
    print("Testing value_iteration_step function...")

    # Create simple test case
    trans, rew, n_s, n_a = create_gridworld_mdp(grid_size=3)

    # Initialize V
    V = np.zeros(n_s)

    try:
        new_V = value_iter_step_func(V, trans, rew, gamma=0.9)

        # Check shape
        assert new_V.shape == (n_s,), f"Wrong shape: {new_V.shape} != {(n_s,)}"

        # Check that values changed (unless all zeros which would be wrong)
        assert not np.allclose(new_V, V), "Values didn't change after iteration"

        # Check that values are finite
        assert np.all(np.isfinite(new_V)), "Values contain NaN or Inf"

        print("\033[92m✓ All tests passed!\033[0m")

    except Exception as e:
        print(f"\033[91m✗ Test failed: {str(e)}\033[0m")
        raise


def extract_policy_test(extract_policy_func):
    """
    Test function for policy extraction from value function.

    Arguments:
    extract_policy_func -- function that extracts policy from V

    Returns:
    None (prints test results)
    """
    print("Testing extract_policy function...")

    # Create simple test case
    trans, rew, n_s, n_a = create_gridworld_mdp(grid_size=3)

    # Create a simple value function (higher values toward goal)
    V = np.arange(n_s, dtype=float)

    try:
        policy = extract_policy_func(V, trans, rew, gamma=0.9)

        # Check shape
        assert policy.shape == (n_s,), f"Wrong shape: {policy.shape} != {(n_s,)}"

        # Check that policy contains valid actions
        assert np.all((policy >= 0) & (policy < n_a)), "Policy contains invalid actions"

        # Check that policy is integer type
        assert policy.dtype in [np.int32, np.int64, int], "Policy should contain integers"

        print("\033[92m✓ All tests passed!\033[0m")

    except Exception as e:
        print(f"\033[91m✗ Test failed: {str(e)}\033[0m")
        raise


def compare_algorithms_test(pi_results, vi_results):
    """
    Test that Policy Iteration and Value Iteration converge to similar solutions.

    Arguments:
    pi_results -- dictionary with Policy Iteration results
    vi_results -- dictionary with Value Iteration results

    Returns:
    None (prints test results)
    """
    print("Testing algorithm convergence...")

    try:
        # Check that both converged
        assert 'V' in pi_results and 'policy' in pi_results, "PI results missing V or policy"
        assert 'V' in vi_results and 'policy' in vi_results, "VI results missing V or policy"

        # Check that values are close
        max_diff = np.max(np.abs(pi_results['V'] - vi_results['V']))
        assert max_diff < 1e-4, f"Value functions differ too much: {max_diff}"

        # Check that policies are mostly the same (allow some differences due to ties)
        policy_diff = np.sum(pi_results['policy'] != vi_results['policy'])
        total_states = len(pi_results['policy'])
        diff_ratio = policy_diff / total_states
        assert diff_ratio < 0.1, f"Policies differ in {diff_ratio*100:.1f}% of states"

        print("\033[92m✓ All tests passed!\033[0m")
        print(f"  - Max value difference: {max_diff:.8f}")
        print(f"  - Policy agreement: {(1-diff_ratio)*100:.1f}%")

    except Exception as e:
        print(f"\033[91m✗ Test failed: {str(e)}\033[0m")
        raise


def gridworld_environment_test(trans, rew, n_s, n_a, expected_size):
    """
    Test that gridworld environment is created correctly.

    Arguments:
    trans -- transition probability array
    rew -- reward array
    n_s -- number of states
    n_a -- number of actions
    expected_size -- expected grid size

    Returns:
    None (prints test results)
    """
    print(f"Testing GridWorld environment ({expected_size}x{expected_size})...")

    try:
        # Check shapes
        assert trans.shape == (n_s, n_a, n_s), f"Wrong transition shape: {trans.shape}"
        assert rew.shape == (n_s, n_a, n_s), f"Wrong reward shape: {rew.shape}"
        assert n_s == expected_size ** 2, f"Wrong number of states: {n_s}"
        assert n_a == 4, f"Wrong number of actions: {n_a}"

        # Check that transition probabilities sum to 1
        prob_sums = np.sum(trans, axis=2)
        assert np.allclose(prob_sums, 1.0), "Transition probabilities don't sum to 1"

        # Check that probabilities are non-negative
        assert np.all(trans >= 0), "Negative transition probabilities"

        print("\033[92m✓ All tests passed!\033[0m")

    except Exception as e:
        print(f"\033[91m✗ Test failed: {str(e)}\033[0m")
        raise


# ==================== EXERCISE TEMPLATES ====================

def test_case_modified_rewards():
    """
    Returns test case parameters for Exercise 1.
    """
    return {
        'grid_size': 4,
        'goal_reward': 1.0,
        'step_reward': -0.1,
        'gamma': 0.99
    }


def test_case_stochastic_gridworld():
    """
    Returns test case parameters for Exercise 2.
    """
    return {
        'grid_size': 4,
        'p_correct': 0.8,
        'gamma': 0.99
    }


def test_case_efficiency_comparison():
    """
    Returns test case parameters for Exercise 3.
    """
    return {
        'grid_sizes': [3, 4, 5, 6],
        'gamma': 0.99
    }


def test_case_multi_goal():
    """
    Returns test case parameters for Exercise 4.
    """
    return {
        'grid_size': 6,
        'goals': [((2, 5), 1.0), ((4, 1), 0.5)],
        'gamma': 0.99
    }


def test_case_sensitivity_analysis():
    """
    Returns test case parameters for Exercise 5.
    """
    return {
        'grid_size': 4,
        'thetas': [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
        'gamma': 0.99
    }


print("✓ dp_utils.py loaded successfully")
print("  Available functions:")
print("  - visualize_policy_and_values()")
print("  - visualize_trajectory()")
print("  - simulate_trajectory()")
print("  - create_gridworld_with_penalties()")
print("  Available tests:")
print("  - policy_evaluation_test()")
print("  - policy_improvement_test()")
print("  - value_iteration_step_test()")
print("  - extract_policy_test()")
print("  - compare_algorithms_test()")
print("  - gridworld_environment_test()")
