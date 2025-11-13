"""
TD Learning Utility Functions and Tests
Author: DeepLearning.AI Style Utilities
"""

import numpy as np
from collections import defaultdict
import gymnasium as gym
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


# ==================== AGENT IMPLEMENTATIONS ====================

class QLearningAgent:
    """
    Q-Learning Agent (Off-Policy)

    Implements the off-policy TD learning algorithm that learns the optimal policy
    while exploring with an epsilon-greedy policy.
    """

    def __init__(self, n_actions: int, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize Q-Learning Agent

        Arguments:
        n_actions -- number of actions in the environment
        alpha -- learning rate (step size)
        gamma -- discount factor
        epsilon -- initial exploration rate
        epsilon_decay -- decay rate for epsilon
        epsilon_min -- minimum epsilon value
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy

        Arguments:
        state -- current state
        training -- whether in training mode (enables exploration)

        Returns:
        action -- selected action (0 to n_actions-1)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action: int, reward: float, next_state, done: bool):
        """
        Q-Learning update rule: Q(s,a) <- Q(s,a) + alpha[r + gamma*max(Q(s',a')) - Q(s,a)]

        Arguments:
        state -- current state
        action -- action taken
        reward -- reward received
        next_state -- resulting state
        done -- whether episode terminated
        """
        current_q = self.Q[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])

        td_error = target_q - current_q
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class SARSAAgent:
    """
    SARSA Agent (On-Policy)

    Implements the on-policy TD learning algorithm that learns about the policy
    being followed during exploration.
    """

    def __init__(self, n_actions: int, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize SARSA Agent

        Arguments:
        n_actions -- number of actions in the environment
        alpha -- learning rate (step size)
        gamma -- discount factor
        epsilon -- initial exploration rate
        epsilon_decay -- decay rate for epsilon
        epsilon_min -- minimum epsilon value
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy

        Arguments:
        state -- current state
        training -- whether in training mode (enables exploration)

        Returns:
        action -- selected action (0 to n_actions-1)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action: int, reward: float, next_state, next_action: int, done: bool):
        """
        SARSA update rule: Q(s,a) <- Q(s,a) + alpha[r + gamma*Q(s',a') - Q(s,a)]

        Arguments:
        state -- current state
        action -- action taken
        reward -- reward received
        next_state -- resulting state
        next_action -- next action to be taken (from actual policy)
        done -- whether episode terminated
        """
        current_q = self.Q[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * self.Q[next_state][next_action]

        td_error = target_q - current_q
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ExpectedSARSAAgent:
    """
    Expected SARSA Agent

    Similar to SARSA but uses expected value over the policy instead of
    the next action's Q-value, providing a balance between Q-Learning and SARSA.
    """

    def __init__(self, n_actions: int, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize Expected SARSA Agent

        Arguments:
        n_actions -- number of actions in the environment
        alpha -- learning rate (step size)
        gamma -- discount factor
        epsilon -- initial exploration rate
        epsilon_decay -- decay rate for epsilon
        epsilon_min -- minimum epsilon value
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy

        Arguments:
        state -- current state
        training -- whether in training mode (enables exploration)

        Returns:
        action -- selected action (0 to n_actions-1)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action: int, reward: float, next_state, done: bool):
        """
        Expected SARSA update: uses expected value over epsilon-greedy policy

        Arguments:
        state -- current state
        action -- action taken
        reward -- reward received
        next_state -- resulting state
        done -- whether episode terminated
        """
        current_q = self.Q[state][action]

        if done:
            target_q = reward
        else:
            # Compute expected value under epsilon-greedy policy
            q_values = self.Q[next_state]
            max_action = np.argmax(q_values)

            # Expected value = (1-epsilon) * max_value + epsilon/n_actions * sum_all
            expected_q = ((1 - self.epsilon) * q_values[max_action] +
                         (self.epsilon / self.n_actions) * np.sum(q_values))

            target_q = reward + self.gamma * expected_q

        td_error = target_q - current_q
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==================== TRAINING FUNCTIONS ====================

def train_q_learning(env, agent: QLearningAgent, n_episodes: int = 500,
                     max_steps: int = 100, verbose: bool = True) -> List[float]:
    """
    Train a Q-Learning agent

    Arguments:
    env -- gymnasium environment
    agent -- QLearningAgent instance
    n_episodes -- number of episodes to train
    max_steps -- maximum steps per episode
    verbose -- whether to print progress

    Returns:
    rewards_history -- list of rewards per episode
    """
    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f'Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.3f} | epsilon: {agent.epsilon:.3f}')

    return rewards_history


def train_sarsa(env, agent: SARSAAgent, n_episodes: int = 500,
                max_steps: int = 100, verbose: bool = True) -> List[float]:
    """
    Train a SARSA agent

    Arguments:
    env -- gymnasium environment
    agent -- SARSAAgent instance
    n_episodes -- number of episodes to train
    max_steps -- maximum steps per episode
    verbose -- whether to print progress

    Returns:
    rewards_history -- list of rewards per episode
    """
    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        action = agent.get_action(state)
        episode_reward = 0

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = agent.get_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            episode_reward += reward
            state = next_state
            action = next_action

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f'Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.3f} | epsilon: {agent.epsilon:.3f}')

    return rewards_history


def train_expected_sarsa(env, agent: ExpectedSARSAAgent, n_episodes: int = 500,
                         max_steps: int = 100, verbose: bool = True) -> List[float]:
    """
    Train an Expected SARSA agent

    Arguments:
    env -- gymnasium environment
    agent -- ExpectedSARSAAgent instance
    n_episodes -- number of episodes to train
    max_steps -- maximum steps per episode
    verbose -- whether to print progress

    Returns:
    rewards_history -- list of rewards per episode
    """
    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f'Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.3f} | epsilon: {agent.epsilon:.3f}')

    return rewards_history


# ==================== TEST FUNCTIONS ====================

def test_q_learning_agent():
    """
    Test basic Q-Learning agent functionality
    """
    agent = QLearningAgent(n_actions=4, alpha=0.1, gamma=0.99)

    # Test initialization
    assert agent.n_actions == 4
    assert agent.alpha == 0.1
    assert agent.gamma == 0.99

    # Test get_action
    action = agent.get_action(0, training=False)
    assert 0 <= action < 4

    # Test update
    agent.update(0, 0, 1.0, 1, done=False)
    assert not np.all(agent.Q[0] == 0)

    print("✓ Q-Learning Agent tests passed")


def test_sarsa_agent():
    """
    Test basic SARSA agent functionality
    """
    agent = SARSAAgent(n_actions=4, alpha=0.1, gamma=0.99)

    # Test initialization
    assert agent.n_actions == 4
    assert agent.alpha == 0.1
    assert agent.gamma == 0.99

    # Test get_action
    action = agent.get_action(0, training=False)
    assert 0 <= action < 4

    # Test update
    agent.update(0, 0, 1.0, 1, 1, done=False)
    assert not np.all(agent.Q[0] == 0)

    print("✓ SARSA Agent tests passed")


def test_expected_sarsa_agent():
    """
    Test basic Expected SARSA agent functionality
    """
    agent = ExpectedSARSAAgent(n_actions=4, alpha=0.1, gamma=0.99)

    # Test initialization
    assert agent.n_actions == 4
    assert agent.alpha == 0.1
    assert agent.gamma == 0.99

    # Test get_action
    action = agent.get_action(0, training=False)
    assert 0 <= action < 4

    # Test update
    agent.update(0, 0, 1.0, 1, done=False)
    assert not np.all(agent.Q[0] == 0)

    print("✓ Expected SARSA Agent tests passed")


def test_td_error_calculation():
    """
    Test TD error calculation

    TD Error = R(t+1) + gamma * V(s') - V(s)
    """
    V_s = 5.0
    V_s_prime = 3.0
    reward = 1.0
    gamma = 0.99

    td_error = reward + gamma * V_s_prime - V_s
    expected_error = 1.0 + 0.99 * 3.0 - 5.0

    assert np.isclose(td_error, expected_error)
    print("✓ TD Error calculation test passed")


def test_epsilon_decay():
    """
    Test epsilon decay mechanism
    """
    agent = QLearningAgent(n_actions=4, epsilon=1.0, epsilon_decay=0.99)
    initial_epsilon = agent.epsilon

    for _ in range(10):
        agent.decay_epsilon()

    assert agent.epsilon < initial_epsilon
    assert agent.epsilon >= agent.epsilon_min
    print("✓ Epsilon decay test passed")


# ==================== VISUALIZATION FUNCTIONS ====================

def plot_training_curves(rewards_dict: Dict[str, List[float]], window: int = 50):
    """
    Plot training curves with moving average

    Arguments:
    rewards_dict -- dictionary of {algorithm_name: rewards_list}
    window -- window size for moving average
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'Q-Learning': 'blue', 'SARSA': 'green', 'Expected SARSA': 'orange'}

    for name, rewards in rewards_dict.items():
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        color = colors.get(name, 'gray')
        ax.plot(range(window-1, len(rewards)), moving_avg, linewidth=2.5,
                label=f'{name} (window={window})', color=color)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('TD Learning Algorithms: Training Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax


def plot_comparison_bars(algorithm_results: Dict[str, float]):
    """
    Plot comparison bar chart

    Arguments:
    algorithm_results -- dictionary of {algorithm_name: final_reward}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(algorithm_results.keys())
    values = list(algorithm_results.values())
    colors = ['blue', 'green', 'orange', 'red'][:len(names)]

    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Algorithm Comparison: Final Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    return fig, ax


if __name__ == '__main__':
    print("Running TD Utils Tests...")
    test_q_learning_agent()
    test_sarsa_agent()
    test_expected_sarsa_agent()
    test_td_error_calculation()
    test_epsilon_decay()
    print("\n✓ All tests passed!")
