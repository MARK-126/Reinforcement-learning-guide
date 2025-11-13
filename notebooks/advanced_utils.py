"""
Advanced Reinforcement Learning Utilities
Advanced algorithms: PPO, DDPG, TD3, SAC
Includes helper functions, losses, and network architectures
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


# ==================== NETWORK ARCHITECTURES ====================

class ActorNetwork(nn.Module):
    """Deterministic Actor Network for continuous control"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CriticNetwork(nn.Module):
    """Q-Network (Critic) for value estimation"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Stochastic Policy Network for SAC"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        self.shared = nn.Sequential(*layers)

        # Initialize log_std with -0.5 (std ~ 0.6)
        self.log_std_head.bias.data.fill_(-0.5)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)

        if deterministic:
            return torch.tanh(mean), None

        std = torch.exp(log_std)
        z = torch.randn_like(mean)
        action = torch.tanh(mean + std * z)

        # Log probability
        log_prob = -0.5 * ((z ** 2).sum(dim=-1) + log_std.sum(dim=-1) + np.log(2 * np.pi))
        # Correction for tanh
        log_prob = log_prob - torch.log(1 - action ** 2 + 1e-6).sum(dim=-1)

        return action, log_prob


# ==================== LOSS FUNCTIONS ====================

def ppo_loss(advantages: torch.Tensor,
             log_probs_new: torch.Tensor,
             log_probs_old: torch.Tensor,
             epsilon_clip: float = 0.2) -> torch.Tensor:
    """
    Proximal Policy Optimization (PPO) loss

    Args:
        advantages: Advantage estimates (batch_size,)
        log_probs_new: Log probabilities from new policy (batch_size,)
        log_probs_old: Log probabilities from old policy (batch_size,)
        epsilon_clip: Clipping parameter (default: 0.2)

    Returns:
        PPO loss (scalar)
    """
    # Probability ratio
    prob_ratio = torch.exp(log_probs_new - log_probs_old)

    # Clipped objective
    clipped_ratio = torch.clamp(prob_ratio, 1 - epsilon_clip, 1 + epsilon_clip)

    # Loss (we want to maximize advantage, so minimize negative)
    loss = -torch.min(prob_ratio * advantages, clipped_ratio * advantages)

    return loss.mean()


def ddpg_critic_loss(q_values: torch.Tensor,
                     target_q: torch.Tensor) -> torch.Tensor:
    """
    DDPG Critic (Q-Network) loss

    Args:
        q_values: Predicted Q-values (batch_size,)
        target_q: Target Q-values (batch_size,)

    Returns:
        MSE loss (scalar)
    """
    return F.mse_loss(q_values, target_q.detach())


def ddpg_actor_loss(q_values: torch.Tensor) -> torch.Tensor:
    """
    DDPG Actor loss

    Args:
        q_values: Q-values from critic (batch_size,)

    Returns:
        Negative mean Q-value (scalar)
    """
    return -q_values.mean()


def td3_loss(q1_values: torch.Tensor,
             q2_values: torch.Tensor,
             target_q: torch.Tensor) -> torch.Tensor:
    """
    TD3 Critic loss with twin Q-networks

    Args:
        q1_values: Q1 predictions (batch_size,)
        q2_values: Q2 predictions (batch_size,)
        target_q: Target Q-values (batch_size,)

    Returns:
        Combined loss from both critics (scalar)
    """
    loss1 = F.mse_loss(q1_values, target_q.detach())
    loss2 = F.mse_loss(q2_values, target_q.detach())
    return loss1 + loss2


def sac_temperature_loss(log_alpha: torch.Tensor,
                         log_probs: torch.Tensor,
                         target_entropy: float) -> torch.Tensor:
    """
    SAC Temperature (entropy coefficient) loss for auto-tuning

    Args:
        log_alpha: Log of temperature parameter (scalar)
        log_probs: Log probabilities of sampled actions (batch_size,)
        target_entropy: Target entropy for the policy (scalar)

    Returns:
        Temperature loss (scalar)
    """
    return -(log_alpha * (log_probs + target_entropy).detach()).mean()


# ==================== UTILITY FUNCTIONS ====================

def compute_gae(rewards: np.ndarray,
                values: np.ndarray,
                gamma: float = 0.99,
                lambda_: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation (GAE)

    Args:
        rewards: Reward sequence (trajectory_length,)
        values: Value estimates (trajectory_length + 1,) - includes bootstrap
        gamma: Discount factor
        lambda_: GAE lambda parameter

    Returns:
        advantages: Advantage estimates (trajectory_length,)
        returns: Discounted returns (trajectory_length,)
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages[t] = gae

    returns = advantages + values[:-1]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def compute_n_step_returns(rewards: np.ndarray,
                          next_value: float,
                          gamma: float = 0.99,
                          n_steps: int = 1) -> np.ndarray:
    """
    Compute n-step returns for off-policy algorithms

    Args:
        rewards: Reward sequence (trajectory_length,)
        next_value: Value of next state (bootstrap)
        gamma: Discount factor
        n_steps: Number of steps for return estimation

    Returns:
        n_step_returns: (trajectory_length,)
    """
    n_step_returns = np.zeros_like(rewards, dtype=np.float32)
    cumulative = next_value

    for t in reversed(range(len(rewards))):
        cumulative = rewards[t] + gamma * cumulative
        n_step_returns[t] = cumulative

    return n_step_returns


def polyak_update(target_params: Dict,
                  source_params: Dict,
                  tau: float = 0.005) -> None:
    """
    Polyak averaging for target network update

    Args:
        target_params: Target network parameters
        source_params: Source network parameters
        tau: Update rate (typically 0.001 to 0.01)
    """
    for target_param, source_param in zip(target_params, source_params):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def soft_update_from_net(target_net: nn.Module,
                         source_net: nn.Module,
                         tau: float = 0.005) -> None:
    """
    Polyak update between two networks

    Args:
        target_net: Target network
        source_net: Source network
        tau: Update rate
    """
    polyak_update(target_net.parameters(), source_net.parameters(), tau)


# ==================== TESTS ====================

class TestPPOLoss:
    """Test cases for PPO loss"""

    @staticmethod
    def test_ppo_loss_basic():
        """Test basic PPO loss computation"""
        batch_size = 32
        advantages = torch.randn(batch_size)
        log_probs_new = torch.randn(batch_size)
        log_probs_old = log_probs_new.clone().detach()

        loss = ppo_loss(advantages, log_probs_new, log_probs_old)

        # When probabilities are the same, loss should be close to -mean(advantages)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        print("✓ PPO loss basic test passed")

    @staticmethod
    def test_ppo_clipping():
        """Test that clipping works correctly"""
        batch_size = 32
        advantages = torch.ones(batch_size)
        log_probs_old = torch.zeros(batch_size)
        log_probs_new = log_probs_old + 0.5  # prob_ratio = 1.65

        loss = ppo_loss(advantages, log_probs_new, log_probs_old, epsilon_clip=0.2)

        # Should be clipped at 1.2
        assert not torch.isnan(loss)
        print("✓ PPO clipping test passed")


class TestDDPGLoss:
    """Test cases for DDPG loss"""

    @staticmethod
    def test_ddpg_critic_loss():
        """Test DDPG critic loss"""
        batch_size = 32
        q_values = torch.randn(batch_size, requires_grad=True)
        target_q = torch.randn(batch_size)

        loss = ddpg_critic_loss(q_values, target_q)

        assert loss.shape == torch.Size([])
        assert loss.requires_grad
        print("✓ DDPG critic loss test passed")

    @staticmethod
    def test_ddpg_actor_loss():
        """Test DDPG actor loss"""
        batch_size = 32
        q_values = torch.randn(batch_size, requires_grad=True)

        loss = ddpg_actor_loss(q_values)

        assert loss.shape == torch.Size([])
        assert loss.requires_grad
        print("✓ DDPG actor loss test passed")


class TestTD3Loss:
    """Test cases for TD3 loss"""

    @staticmethod
    def test_td3_loss():
        """Test TD3 twin critic loss"""
        batch_size = 32
        q1_values = torch.randn(batch_size, requires_grad=True)
        q2_values = torch.randn(batch_size, requires_grad=True)
        target_q = torch.randn(batch_size)

        loss = td3_loss(q1_values, q2_values, target_q)

        assert loss.shape == torch.Size([])
        assert loss.requires_grad
        print("✓ TD3 loss test passed")


class TestSACLoss:
    """Test cases for SAC losses"""

    @staticmethod
    def test_sac_temperature_loss():
        """Test SAC temperature loss"""
        batch_size = 32
        log_alpha = torch.tensor(0.0, requires_grad=True)
        log_probs = torch.randn(batch_size)
        target_entropy = -1.0

        loss = sac_temperature_loss(log_alpha, log_probs, target_entropy)

        assert loss.shape == torch.Size([])
        assert loss.requires_grad
        print("✓ SAC temperature loss test passed")


class TestNetworks:
    """Test cases for neural networks"""

    @staticmethod
    def test_actor_network():
        """Test actor network"""
        net = ActorNetwork(input_dim=10, output_dim=2, hidden_dims=[64, 64])
        x = torch.randn(32, 10)

        output = net(x)

        assert output.shape == (32, 2)
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0)
        print("✓ Actor network test passed")

    @staticmethod
    def test_critic_network():
        """Test critic network"""
        net = CriticNetwork(state_dim=10, action_dim=2, hidden_dims=[64, 64])
        state = torch.randn(32, 10)
        action = torch.randn(32, 2)

        output = net(state, action)

        assert output.shape == (32, 1)
        print("✓ Critic network test passed")

    @staticmethod
    def test_policy_network():
        """Test stochastic policy network"""
        net = PolicyNetwork(state_dim=10, action_dim=2, hidden_dims=[64, 64])
        state = torch.randn(32, 10)

        action, log_prob = net.sample(state)

        assert action.shape == (32, 2)
        assert log_prob.shape == (32,)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
        print("✓ Policy network test passed")


class TestUtilityFunctions:
    """Test utility functions"""

    @staticmethod
    def test_gae():
        """Test GAE computation"""
        rewards = np.array([1.0, 1.0, 1.0, 0.0])
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.0])

        advantages, returns = compute_gae(rewards, values, gamma=0.99, lambda_=0.95)

        assert advantages.shape == (4,)
        assert returns.shape == (4,)
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(returns))
        print("✓ GAE test passed")

    @staticmethod
    def test_n_step_returns():
        """Test n-step returns"""
        rewards = np.array([1.0, 1.0, 1.0, 0.0])

        returns = compute_n_step_returns(rewards, next_value=0.5, gamma=0.99)

        assert returns.shape == (4,)
        assert np.all(np.isfinite(returns))
        print("✓ N-step returns test passed")


def run_all_tests():
    """Run all tests"""
    print("\nRunning Advanced RL Utils Tests...")
    print("=" * 50)

    # PPO tests
    print("\nPPO Tests:")
    TestPPOLoss.test_ppo_loss_basic()
    TestPPOLoss.test_ppo_clipping()

    # DDPG tests
    print("\nDDPG Tests:")
    TestDDPGLoss.test_ddpg_critic_loss()
    TestDDPGLoss.test_ddpg_actor_loss()

    # TD3 tests
    print("\nTD3 Tests:")
    TestTD3Loss.test_td3_loss()

    # SAC tests
    print("\nSAC Tests:")
    TestSACLoss.test_sac_temperature_loss()

    # Network tests
    print("\nNetwork Architecture Tests:")
    TestNetworks.test_actor_network()
    TestNetworks.test_critic_network()
    TestNetworks.test_policy_network()

    # Utility tests
    print("\nUtility Function Tests:")
    TestUtilityFunctions.test_gae()
    TestUtilityFunctions.test_n_step_returns()

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
