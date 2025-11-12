"""
Plotting utilities for visualizing RL training results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_rewards(rewards: List[float], 
                 window: int = 100,
                 title: str = "Training Rewards",
                 save_path: Optional[str] = None):
    """
    Plot rewards over episodes with moving average
    
    Args:
        rewards: List of episode rewards
        window: Window size for moving average
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw rewards
    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(rewards)), moving_avg, 
               color='red', linewidth=2, label=f'Moving Average ({window} eps)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_metrics(metrics: Dict[str, List[float]],
                          window: int = 50,
                          title: str = "Training Metrics",
                          save_path: Optional[str] = None):
    """
    Plot multiple training metrics in subplots
    
    Args:
        metrics: Dictionary with metric names and values
        window: Window size for moving average
        title: Overall title
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        episodes = np.arange(len(values))
        
        # Plot raw values
        ax.plot(episodes, values, alpha=0.3, label=name)
        
        # Plot moving average
        if len(values) >= window:
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(values)), moving_avg,
                   linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_value_function(V: np.ndarray,
                       shape: Tuple[int, int],
                       title: str = "Value Function",
                       save_path: Optional[str] = None):
    """
    Plot value function as heatmap (for GridWorld-like environments)
    
    Args:
        V: Value function (can be 1D or 2D)
        shape: Shape to reshape V if it's 1D
        title: Plot title
        save_path: Path to save figure
    """
    if V.ndim == 1:
        V = V.reshape(shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(V, cmap='RdYlGn', interpolation='nearest')
    
    # Add values to cells
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            text = ax.text(j, i, f'{V[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(V.shape[1]))
    ax.set_yticks(np.arange(V.shape[0]))
    
    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_policy(policy: np.ndarray,
               shape: Tuple[int, int],
               action_names: List[str] = ['↑', '→', '↓', '←'],
               title: str = "Policy",
               save_path: Optional[str] = None):
    """
    Plot policy as arrows in a grid
    
    Args:
        policy: Policy array (action indices)
        shape: Grid shape
        action_names: Names/symbols for actions
        title: Plot title
        save_path: Path to save figure
    """
    if policy.ndim == 1:
        policy = policy.reshape(shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create grid
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(-0.5, shape[0] - 0.5)
    ax.set_aspect('equal')
    
    # Draw grid lines
    for i in range(shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)
    
    # Draw policy arrows
    for i in range(shape[0]):
        for j in range(shape[1]):
            action = policy[i, j]
            if action < len(action_names):
                ax.text(j, i, action_names[action],
                       ha='center', va='center', fontsize=20)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_q_values(Q: np.ndarray,
                 state: int,
                 action_names: Optional[List[str]] = None,
                 title: Optional[str] = None,
                 save_path: Optional[str] = None):
    """
    Plot Q-values for a specific state as bar chart
    
    Args:
        Q: Q-value array
        state: State index
        action_names: Names for actions
        title: Plot title
        save_path: Path to save figure
    """
    q_values = Q[state]
    n_actions = len(q_values)
    
    if action_names is None:
        action_names = [f'Action {i}' for i in range(n_actions)]
    
    if title is None:
        title = f'Q-values for State {state}'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(action_names, q_values, color='steelblue', edgecolor='black')
    
    # Highlight best action
    best_action = np.argmax(q_values)
    bars[best_action].set_color('green')
    bars[best_action].set_edgecolor('darkgreen')
    bars[best_action].set_linewidth(2)
    
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Q-value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, q_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}',
               ha='center', va='bottom' if value >= 0 else 'top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_learning_curves(results: Dict[str, Dict[str, List[float]]],
                         metric: str = 'reward',
                         window: int = 50,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None):
    """
    Compare learning curves of multiple algorithms
    
    Args:
        results: Dictionary of {algorithm_name: {metric: values}}
        metric: Which metric to plot
        window: Moving average window
        title: Plot title
        save_path: Path to save figure
    """
    if title is None:
        title = f'Comparison of {metric.capitalize()}'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for alg_name, metrics in results.items():
        if metric not in metrics:
            continue
        
        values = metrics[metric]
        episodes = np.arange(len(values))
        
        # Plot moving average
        if len(values) >= window:
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(values)), moving_avg,
                   linewidth=2, label=alg_name)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_episode_length(episode_lengths: List[int],
                       window: int = 50,
                       title: str = "Episode Length",
                       save_path: Optional[str] = None):
    """
    Plot episode lengths over training
    
    Args:
        episode_lengths: List of episode lengths
        window: Moving average window
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episodes = np.arange(len(episode_lengths))
    ax.plot(episodes, episode_lengths, alpha=0.3, color='blue', label='Episode Length')
    
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(episode_lengths)), moving_avg,
               color='red', linewidth=2, label=f'MA({window})')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
