"""
Deep Reinforcement Learning
============================

Algoritmos de RL con deep learning:
- DQN y variantes (dqn/)
- Policy Gradient methods (policy_gradient/)
- Algoritmos avanzados (advanced/): PPO, DDPG, TD3, SAC

Módulos disponibles:
-------------------
- dqn: Deep Q-Networks y variantes (Double DQN, Dueling DQN)
- policy_gradient: REINFORCE, Actor-Critic (A2C)
- advanced: Algoritmos SOTA (PPO, DDPG, TD3, SAC)

Quick Start:
-----------
```python
# PPO para ambientes discretos/continuos
from advanced import PPOAgent
agent = PPOAgent(state_dim=4, action_dim=2, continuous=False)

# SAC para control continuo (SOTA)
from advanced import SACAgent
agent = SACAgent(state_dim=3, action_dim=1, auto_tune=True)

# TD3 para control continuo determinista
from advanced import TD3Agent
agent = TD3Agent(state_dim=3, action_dim=1, max_action=2.0)
```
"""

__all__ = ['dqn', 'policy_gradient', 'advanced']
