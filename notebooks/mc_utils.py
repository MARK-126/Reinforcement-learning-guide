"""
Utility functions and test cases for Monte Carlo Methods in Reinforcement Learning
Author: DeepLearning.AI style tutorial
"""

import numpy as np
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# HELPER FUNCTIONS FOR MC PREDICTION
# ============================================================================

def calculate_returns(rewards, gamma=0.99):
    """
    Calculate returns for each step in an episode.

    Arguments:
    rewards -- list of rewards [R1, R2, R3, ...]
    gamma -- discount factor

    Returns:
    returns -- list of returns [G0, G1, G2, ...]
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def first_visit_mc_prediction(env, policy, num_episodes, max_steps, gamma=0.99):
    """
    First-Visit Monte Carlo Prediction.

    Arguments:
    env -- environment with reset() and step() methods
    policy -- function that returns action given state
    num_episodes -- number of episodes to generate
    max_steps -- maximum steps per episode
    gamma -- discount factor

    Returns:
    V -- dictionary of state values
    visit_counts -- dictionary of visit counts per state
    """
    V = defaultdict(float)
    visit_counts = defaultdict(int)
    returns = defaultdict(list)

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        rewards = []
        visited_states = set()

        # Generate episode
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append(state)
            rewards.append(reward)

            if done:
                trajectory.append(next_state)
                break
            state = next_state

        # Calculate returns and update V (first-visit only)
        episode_returns = calculate_returns(rewards, gamma)
        for t, (s, G) in enumerate(zip(trajectory[:-1], episode_returns)):
            if s not in visited_states:
                returns[s].append(G)
                visit_counts[s] += 1
                V[s] = np.mean(returns[s])
                visited_states.add(s)

    return dict(V), dict(visit_counts)


def every_visit_mc_prediction(env, policy, num_episodes, max_steps, gamma=0.99):
    """
    Every-Visit Monte Carlo Prediction.

    Arguments:
    env -- environment with reset() and step() methods
    policy -- function that returns action given state
    num_episodes -- number of episodes to generate
    max_steps -- maximum steps per episode
    gamma -- discount factor

    Returns:
    V -- dictionary of state values
    visit_counts -- dictionary of visit counts per state
    """
    V = defaultdict(float)
    visit_counts = defaultdict(int)
    returns = defaultdict(list)

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        rewards = []

        # Generate episode
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append(state)
            rewards.append(reward)

            if done:
                trajectory.append(next_state)
                break
            state = next_state

        # Calculate returns and update V (every visit)
        episode_returns = calculate_returns(rewards, gamma)
        for t, (s, G) in enumerate(zip(trajectory[:-1], episode_returns)):
            returns[s].append(G)
            visit_counts[s] += 1
            V[s] = np.mean(returns[s])

    return dict(V), dict(visit_counts)


def mc_control_on_policy(env, num_episodes, epsilon=0.1, max_steps=100, gamma=0.99, alpha=None):
    """
    Monte Carlo Control (On-Policy).

    Arguments:
    env -- environment with reset() and step() methods
    num_episodes -- number of episodes to generate
    epsilon -- exploration rate (epsilon-greedy)
    max_steps -- maximum steps per episode
    gamma -- discount factor
    alpha -- learning rate (if None, uses incremental averaging)

    Returns:
    Q -- dictionary of Q-values Q[state][action]
    policy -- optimal policy derived from Q
    """
    Q = defaultdict(lambda: defaultdict(float))
    returns = defaultdict(lambda: defaultdict(list))

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        actions_taken = []
        rewards = []

        # Generate episode following epsilon-greedy policy
        for step in range(max_steps):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample() if hasattr(env.action_space, 'sample') else np.random.randint(0, 4)
            else:
                if len(Q[state]) == 0:
                    action = env.action_space.sample() if hasattr(env.action_space, 'sample') else np.random.randint(0, 4)
                else:
                    action = max(Q[state].items(), key=lambda x: x[1])[0]

            trajectory.append(state)
            actions_taken.append(action)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if done:
                trajectory.append(next_state)
                break
            state = next_state

        # Update Q-values (first-visit)
        episode_returns = calculate_returns(rewards, gamma)
        visited = set()
        for t, (s, a, G) in enumerate(zip(trajectory[:-1], actions_taken, episode_returns)):
            if (s, a) not in visited:
                returns[s][a].append(G)
                if alpha is None:
                    Q[s][a] = np.mean(returns[s][a])
                else:
                    Q[s][a] += alpha * (G - Q[s][a])
                visited.add((s, a))

    # Extract policy
    policy = {}
    for state in Q:
        if len(Q[state]) > 0:
            policy[state] = max(Q[state].items(), key=lambda x: x[1])[0]

    return dict(Q), policy


def mc_control_off_policy(env, num_episodes, epsilon=0.3, max_steps=100, gamma=0.99):
    """
    Monte Carlo Control (Off-Policy with Importance Sampling).

    Arguments:
    env -- environment with reset() and step() methods
    num_episodes -- number of episodes to generate
    epsilon -- exploration rate for behavior policy
    max_steps -- maximum steps per episode
    gamma -- discount factor

    Returns:
    Q -- dictionary of Q-values Q[state][action]
    policy -- optimal target policy
    importance_ratios -- list of importance ratios per episode
    """
    Q = defaultdict(lambda: defaultdict(float))
    C = defaultdict(lambda: defaultdict(float))  # Cumulative weights
    importance_ratios = []

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        actions_taken = []
        rewards = []

        # Generate episode using behavior policy (epsilon-greedy with high epsilon)
        for step in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample() if hasattr(env.action_space, 'sample') else np.random.randint(0, 4)
            else:
                if len(Q[state]) == 0:
                    action = env.action_space.sample() if hasattr(env.action_space, 'sample') else np.random.randint(0, 4)
                else:
                    action = max(Q[state].items(), key=lambda x: x[1])[0]

            trajectory.append(state)
            actions_taken.append(action)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if done:
                trajectory.append(next_state)
                break
            state = next_state

        # Off-policy update with importance sampling
        G = 0
        W = 1
        episode_ratio = 1

        for t in reversed(range(len(trajectory) - 1)):
            s = trajectory[t]
            a = actions_taken[t]
            r = rewards[t]

            G = r + gamma * G
            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])

            # Check if action is greedy (target policy)
            best_action = max(Q[s].items(), key=lambda x: x[1])[0] if len(Q[s]) > 0 else 0
            if a != best_action:
                break

            # Update importance ratio
            num_actions = 4  # Assuming 4 actions
            W *= (1.0 / epsilon) if np.random.random() < epsilon else 1.0
            episode_ratio *= W

        importance_ratios.append(episode_ratio)

    # Extract policy
    policy = {}
    for state in Q:
        if len(Q[state]) > 0:
            policy[state] = max(Q[state].items(), key=lambda x: x[1])[0]

    return dict(Q), policy, importance_ratios


# ============================================================================
# TEST CASES
# ============================================================================

def first_visit_mc_prediction_test(func):
    """Test first-visit MC prediction implementation"""
    from collections import namedtuple

    # Create simple test environment
    class SimpleEnv:
        def __init__(self):
            self.current_state = 0

        def reset(self):
            self.current_state = 0
            return 0

        def step(self, action):
            if self.current_state == 0:
                next_state = 1
                reward = 1
                done = False
            elif self.current_state == 1:
                next_state = 2
                reward = 1
                done = False
            else:
                next_state = 0
                reward = 0
                done = True
            self.current_state = next_state
            return next_state, reward, done, {}

    env = SimpleEnv()
    policy = lambda s: 0  # Always take action 0

    V, counts = func(env, policy, num_episodes=100, max_steps=10, gamma=0.99)

    assert isinstance(V, dict), "V should be a dictionary"
    assert isinstance(counts, dict), "counts should be a dictionary"
    assert len(V) > 0, "V should not be empty"

    print("First-Visit MC Prediction test passed!")


def every_visit_mc_prediction_test(func):
    """Test every-visit MC prediction implementation"""

    class SimpleEnv:
        def __init__(self):
            self.current_state = 0

        def reset(self):
            self.current_state = 0
            return 0

        def step(self, action):
            if self.current_state == 0:
                next_state = 1
                reward = 1
                done = False
            elif self.current_state == 1:
                next_state = 2
                reward = 1
                done = False
            else:
                next_state = 0
                reward = 0
                done = True
            self.current_state = next_state
            return next_state, reward, done, {}

    env = SimpleEnv()
    policy = lambda s: 0

    V, counts = func(env, policy, num_episodes=100, max_steps=10, gamma=0.99)

    assert isinstance(V, dict), "V should be a dictionary"
    assert isinstance(counts, dict), "counts should be a dictionary"

    print("Every-Visit MC Prediction test passed!")


def mc_control_on_policy_test(func):
    """Test MC control on-policy implementation"""

    class SimpleEnv:
        def __init__(self):
            self.current_state = 0
            self.action_space = type('ActionSpace', (), {'sample': lambda: 0})()

        def reset(self):
            self.current_state = 0
            return 0

        def step(self, action):
            if self.current_state < 2:
                next_state = self.current_state + 1
                reward = 1
                done = False
            else:
                next_state = 0
                reward = 0
                done = True
            self.current_state = next_state
            return next_state, reward, done, {}

    env = SimpleEnv()
    Q, policy = func(env, num_episodes=50, epsilon=0.1, max_steps=10, gamma=0.99)

    assert isinstance(Q, dict), "Q should be a dictionary"
    assert isinstance(policy, dict), "policy should be a dictionary"

    print("MC Control On-Policy test passed!")


def mc_control_off_policy_test(func):
    """Test MC control off-policy implementation"""

    class SimpleEnv:
        def __init__(self):
            self.current_state = 0
            self.action_space = type('ActionSpace', (), {'sample': lambda: 0})()

        def reset(self):
            self.current_state = 0
            return 0

        def step(self, action):
            if self.current_state < 2:
                next_state = self.current_state + 1
                reward = 1
                done = False
            else:
                next_state = 0
                reward = 0
                done = True
            self.current_state = next_state
            return next_state, reward, done, {}

    env = SimpleEnv()
    Q, policy, ratios = func(env, num_episodes=50, epsilon=0.3, max_steps=10, gamma=0.99)

    assert isinstance(Q, dict), "Q should be a dictionary"
    assert isinstance(policy, dict), "policy should be a dictionary"
    assert isinstance(ratios, list), "ratios should be a list"

    print("MC Control Off-Policy test passed!")


def importance_sampling_test(func):
    """Test importance sampling implementation"""

    episode = [(0, 0, 1), (1, 0, 1), (2, 1, -1)]  # (state, action, reward)
    behavior_prob = 0.2  # Behavior policy probability for selected action
    target_prob = 0.7    # Target policy probability for selected action

    ratio = func(episode, behavior_prob, target_prob)

    assert isinstance(ratio, (float, np.ndarray)), "Ratio should be numeric"

    print("Importance Sampling test passed!")


# ============================================================================
# TEST CASE DATA GENERATORS
# ============================================================================

def first_visit_mc_prediction_test_case():
    """Generate test case for first-visit MC prediction"""
    return None  # Simplified for now


def every_visit_mc_prediction_test_case():
    """Generate test case for every-visit MC prediction"""
    return None


def mc_control_on_policy_test_case():
    """Generate test case for MC control on-policy"""
    return None


def mc_control_off_policy_test_case():
    """Generate test case for MC control off-policy"""
    return None


def importance_sampling_test_case():
    """Generate test case for importance sampling"""
    episode = [(0, 0, 1), (1, 0, 1), (2, 1, -1)]
    return episode, 0.2, 0.7
