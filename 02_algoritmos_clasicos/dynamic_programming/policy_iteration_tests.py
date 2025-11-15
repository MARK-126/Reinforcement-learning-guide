"""
Test functions for Policy Iteration Exercise
"""

import numpy as np
import gymnasium as gym


def initialize_policy_test(initialize_policy):
    """Test initialize_policy function"""
    print("Testing initialize_policy...")

    # Test case 1: Small environment
    n_states, n_actions = 4, 2
    policy = initialize_policy(n_states, n_actions)

    # Check type and shape
    assert isinstance(policy, np.ndarray), "Policy should be numpy array"
    assert policy.shape == (n_states, n_actions), f"Expected shape {(n_states, n_actions)}, got {policy.shape}"

    # Check probabilities sum to 1 for each state
    for s in range(n_states):
        prob_sum = np.sum(policy[s])
        assert np.isclose(prob_sum, 1.0), f"State {s}: probabilities sum to {prob_sum}, expected 1.0"

    # Check uniform distribution
    expected_prob = 1.0 / n_actions
    assert np.allclose(policy, expected_prob), f"Expected uniform probability {expected_prob}"

    print(f"  Policy shape: {policy.shape}")
    print(f"  Sample probabilities for state 0: {policy[0]}")
    print(f"  All probabilities equal to 1/{n_actions}: {np.allclose(policy, expected_prob)}")

    # Test case 2: Larger environment
    n_states, n_actions = 16, 4
    policy = initialize_policy(n_states, n_actions)
    assert policy.shape == (n_states, n_actions)
    assert np.allclose(policy, 0.25)

    print("\n✅ All tests passed!")


def policy_evaluation_step_test(policy_evaluation_step):
    """Test policy_evaluation_step function"""
    print("Testing policy_evaluation_step...")

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Create a simple uniform policy
    policy = np.ones((n_states, n_actions)) / n_actions
    V = np.zeros(n_states)
    gamma = 0.99

    # Test state 0 (starting state)
    new_value = policy_evaluation_step(env, V, policy, state=0, gamma=gamma)

    # Check type
    assert isinstance(new_value, (float, np.floating)), "Return value should be a float"

    # For uniform policy starting from zero V, the value should be non-negative
    assert new_value >= 0, f"Value should be non-negative, got {new_value}"

    print(f"  State 0 value (from zero V): {new_value:.6f}")

    # Test with non-zero V
    V = np.random.rand(n_states)
    new_value = policy_evaluation_step(env, V, policy, state=0, gamma=gamma)
    print(f"  State 0 value (from random V): {new_value:.6f}")

    # Test deterministic policy
    deterministic_policy = np.zeros((n_states, n_actions))
    deterministic_policy[:, 2] = 1.0  # Always go right
    V = np.zeros(n_states)
    new_value = policy_evaluation_step(env, V, deterministic_policy, state=0, gamma=gamma)
    print(f"  State 0 value (deterministic right policy): {new_value:.6f}")

    print("\n✅ All tests passed!")


def policy_evaluation_test(policy_evaluation):
    """Test policy_evaluation function"""
    print("Testing policy_evaluation...")

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Test with uniform random policy
    policy = np.ones((n_states, n_actions)) / n_actions
    V, iterations = policy_evaluation(env, policy, gamma=0.99, theta=1e-8)

    # Check return types
    assert isinstance(V, np.ndarray), "V should be numpy array"
    assert isinstance(iterations, int), "iterations should be int"

    # Check shape
    assert V.shape == (n_states,), f"Expected shape {(n_states,)}, got {V.shape}"

    # Check convergence
    assert iterations > 0, "Should take at least 1 iteration"
    assert iterations < 1000, f"Too many iterations: {iterations}"

    # Check terminal states have value 0
    # In FrozenLake, states 5, 7, 11, 12, 15 are terminal (holes or goal)
    # Actually in gym FrozenLake, only goal (15) gives reward, holes end episode

    print(f"  Converged in {iterations} iterations")
    print(f"  Value function sample:")
    print(f"    V[0] (start): {V[0]:.6f}")
    print(f"    V[15] (goal): {V[15]:.6f}")

    # Test with deterministic policy (always right)
    deterministic_policy = np.zeros((n_states, n_actions))
    deterministic_policy[:, 2] = 1.0
    V2, iterations2 = policy_evaluation(env, deterministic_policy, gamma=0.99)

    print(f"  Deterministic policy converged in {iterations2} iterations")
    print(f"    V[0]: {V2[0]:.6f}")

    print("\n✅ All tests passed!")


def policy_improvement_test(policy_improvement):
    """Test policy_improvement function"""
    print("Testing policy_improvement...")

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Start with a simple value function
    V = np.random.rand(n_states)
    gamma = 0.99

    new_policy = policy_improvement(env, V, gamma=gamma)

    # Check type and shape
    assert isinstance(new_policy, np.ndarray), "Policy should be numpy array"
    assert new_policy.shape == (n_states, n_actions), f"Expected shape {(n_states, n_actions)}, got {new_policy.shape}"

    # Check it's a valid probability distribution
    for s in range(n_states):
        prob_sum = np.sum(new_policy[s])
        assert np.isclose(prob_sum, 1.0), f"State {s}: probabilities sum to {prob_sum}"

    # Check it's deterministic (one action has probability 1.0)
    for s in range(n_states):
        max_prob = np.max(new_policy[s])
        assert np.isclose(max_prob, 1.0), f"State {s}: should be deterministic, max prob = {max_prob}"

        # Count actions with probability 1.0
        n_optimal = np.sum(np.isclose(new_policy[s], 1.0))
        assert n_optimal == 1, f"State {s}: should have exactly 1 optimal action, found {n_optimal}"

    print(f"  Policy shape: {new_policy.shape}")
    print(f"  Sample actions (argmax):")
    action_map = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
    for s in [0, 1, 2, 3]:
        action = np.argmax(new_policy[s])
        print(f"    State {s}: {action_map[action]}")

    print("\n✅ All tests passed!")


def policy_iteration_test(policy_iteration):
    """Test complete policy_iteration function"""
    print("Testing policy_iteration...")

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Run policy iteration
    policy, V, iterations = policy_iteration(env, gamma=0.99, theta=1e-8)

    # Check return types
    assert isinstance(policy, np.ndarray), "Policy should be numpy array"
    assert isinstance(V, np.ndarray), "V should be numpy array"
    assert isinstance(iterations, int), "Iterations should be int"

    # Check shapes
    assert policy.shape == (n_states, n_actions), f"Expected policy shape {(n_states, n_actions)}"
    assert V.shape == (n_states,), f"Expected V shape {(n_states,)}"

    # Check convergence
    assert iterations > 0, "Should take at least 1 iteration"
    assert iterations < 100, f"Too many iterations: {iterations}"

    # Check policy is deterministic
    for s in range(n_states):
        assert np.isclose(np.sum(policy[s]), 1.0), f"State {s}: invalid probability distribution"
        assert np.isclose(np.max(policy[s]), 1.0), f"State {s}: policy should be deterministic"

    # Check value function is reasonable
    assert V[15] >= V[0], "Goal state should have higher or equal value than start"
    assert np.all(V >= 0) or np.all(V <= 0), "Values should be consistent in sign"

    print(f"  Converged in {iterations} policy iterations")
    print(f"  Value function:")
    print(f"    V[0] (start): {V[0]:.6f}")
    print(f"    V[15] (goal): {V[15]:.6f}")

    print(f"\n  Optimal policy (first row):")
    action_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    for s in range(4):
        action = np.argmax(policy[s])
        print(f"    State {s}: {action_map[action]}")

    # Test that running again converges to same solution
    policy2, V2, iterations2 = policy_iteration(env, gamma=0.99, theta=1e-8)
    assert np.allclose(V, V2, atol=1e-5), "Multiple runs should converge to same value function"

    print("\n✅ All tests passed!")
