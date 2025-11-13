"""
Policy Gradient Methods for Deep Reinforcement Learning
=========================================================

Este módulo implementa algoritmos de Policy Gradient que aprenden directamente
una política paramétrica mediante gradiente ascendente en la función objetivo.

Algoritmos implementados:
-------------------------

1. REINFORCE (Monte Carlo Policy Gradient)
   - Algoritmo básico de policy gradient usando retornos de Monte Carlo
   - Variante con baseline para reducir varianza
   - Soporte para espacios discretos y continuos
   - Paper: Williams (1992)

2. Advantage Actor-Critic (A2C)
   - Combina policy gradient con value-based methods
   - Actor aprende la política, Critic estima el valor
   - Soporte para GAE (Generalized Advantage Estimation)
   - Menor varianza que REINFORCE
   - Papers: Mnih et al. (2016), Schulman et al. (2016)

Características principales:
----------------------------
- Implementaciones con PyTorch
- Soporte para espacios de acción discretos y continuos
- Regularización de entropía para mejorar exploración
- Baseline/Critic para reducir varianza
- GAE para mejor estimación de ventajas
- Gradient clipping para estabilidad
- Type hints y documentación completa en español
- Ejemplos de uso con ambientes gymnasium

Uso básico:
-----------
```python
from policy_gradient import REINFORCEAgent, A2CAgent
import gymnasium as gym

# REINFORCE
env = gym.make('CartPole-v1')
agent = REINFORCEAgent(
    state_dim=4,
    action_dim=2,
    use_baseline=True
)
agent.train(env, n_episodes=500)

# A2C
agent_a2c = A2CAgent(
    state_dim=4,
    action_dim=2,
    use_gae=True,
    gae_lambda=0.95
)
agent_a2c.train(env, n_episodes=500)
```

Comparación de algoritmos:
--------------------------
REINFORCE:
  + Simple y fácil de implementar
  + Convergencia garantizada (localmente)
  - Alta varianza
  - Requiere episodios completos
  - Menos eficiente en muestras

A2C:
  + Menor varianza que REINFORCE
  + Más eficiente en muestras
  + Puede usar n-step updates
  + GAE mejora estimación de ventajas
  - Más complejo (dos redes)
  - Puede sufrir de sesgo del value function

Referencias:
------------
- Williams, R. J. (1992). "Simple statistical gradient-following algorithms
  for connectionist reinforcement learning"
- Mnih, V. et al. (2016). "Asynchronous methods for deep reinforcement learning"
- Schulman, J. et al. (2016). "High-dimensional continuous control using
  generalized advantage estimation"

Autor: MARK-126
"""

from .reinforce import (
    REINFORCEAgent,
    PolicyNetwork as REINFORCEPolicyNetwork,
    ValueNetwork as REINFORCEValueNetwork,
    plot_training_results as plot_reinforce_results,
    evaluate_agent as evaluate_reinforce
)

from .actor_critic import (
    A2CAgent,
    ActorNetwork,
    CriticNetwork,
    plot_training_results as plot_a2c_results,
    evaluate_agent as evaluate_a2c
)

__all__ = [
    # REINFORCE
    'REINFORCEAgent',
    'REINFORCEPolicyNetwork',
    'REINFORCEValueNetwork',
    'plot_reinforce_results',
    'evaluate_reinforce',

    # A2C
    'A2CAgent',
    'ActorNetwork',
    'CriticNetwork',
    'plot_a2c_results',
    'evaluate_a2c',
]

__version__ = '1.0.0'
