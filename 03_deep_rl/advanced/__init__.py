"""
Advanced Deep Reinforcement Learning Algorithms
================================================

Este módulo contiene implementaciones de algoritmos avanzados de Deep RL
que representan el estado del arte en control continuo y aprendizaje por refuerzo.

Algoritmos incluidos:
---------------------

1. **PPO (Proximal Policy Optimization)**:
   - Algoritmo on-policy más popular y confiable
   - Clipped surrogate objective para updates seguros
   - GAE para estimación de ventajas
   - Funciona en ambientes discretos y continuos

2. **DDPG (Deep Deterministic Policy Gradient)**:
   - Actor-critic off-policy para control continuo
   - Política determinista + ruido para exploración
   - Ornstein-Uhlenbeck o ruido Gaussiano
   - Base para TD3 y SAC

3. **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**:
   - Mejora sobre DDPG con tres innovaciones clave
   - Twin critics para reducir sobreestimación
   - Delayed policy updates para estabilidad
   - Target policy smoothing para robustez

4. **SAC (Soft Actor-Critic)**:
   - Estado del arte para control continuo
   - Maximum entropy framework
   - Automatic temperature tuning
   - Política estocástica con reparameterization trick

Comparación de algoritmos:
--------------------------

+----------+----------+-------------+-----------+----------------+
| Algoritmo| On/Off   | Acción      | Política  | Características|
+----------+----------+-------------+-----------+----------------+
| PPO      | On-policy| Disc/Cont   | Stochastic| Simple, robusto|
| DDPG     | Off-policy| Continuous | Determ.   | Sample efficient|
| TD3      | Off-policy| Continuous | Determ.   | Mejora DDPG    |
| SAC      | Off-policy| Continuous | Stochastic| SOTA, max-ent  |
+----------+----------+-------------+-----------+----------------+

Cuándo usar cada algoritmo:
---------------------------

- **PPO**: Primera opción para la mayoría de problemas. Confiable y fácil de tunear.
  Ideal para: robótica, juegos, problemas diversos.

- **DDPG**: Control continuo básico. Buena base para aprender.
  Ideal para: problemas simples, prototipado rápido.

- **TD3**: Control continuo con mejor estabilidad que DDPG.
  Ideal para: cuando DDPG es inestable, problemas que requieren determinismo.

- **SAC**: Estado del arte para control continuo.
  Ideal para: robótica real, problemas complejos, cuando sample efficiency importa.

Ejemplo de uso:
---------------

```python
import gymnasium as gym
from advanced.ppo import PPOAgent
from advanced.td3 import TD3Agent
from advanced.sac import SACAgent

# PPO para CartPole (discreto)
env = gym.make('CartPole-v1')
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    continuous=False
)
agent.train(env, n_episodes=300)

# TD3 para Pendulum (continuo)
env = gym.make('Pendulum-v1')
agent = TD3Agent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    max_action=float(env.action_space.high[0])
)
agent.train(env, n_episodes=200)

# SAC para control continuo (SOTA)
env = gym.make('Pendulum-v1')
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    auto_tune=True  # Automatic temperature tuning
)
agent.train(env, n_episodes=150)
```

Tips de entrenamiento:
----------------------

1. **Warmup period**: Usa exploración aleatoria inicial para llenar replay buffer
2. **Hyperparameter tuning**: Empieza con defaults, ajusta learning rates si hay inestabilidad
3. **Network size**: 256-256 funciona bien para la mayoría de problemas
4. **Batch size**: Más grande = más estable pero más lento (128-256 típico)
5. **Buffer size**: 1M para problemas complejos, 100K para simples
6. **Evaluation**: Evalúa con política determinista/sin ruido

Referencias:
------------

Papers originales:
- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- DDPG: "Continuous Control with Deep RL" (Lillicrap et al., 2016)
- TD3: "Addressing Function Approximation Error" (Fujimoto et al., 2018)
- SAC: "Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2019)

Autor: MARK-126
"""

# Import main agent classes
from .ppo import PPOAgent, ActorNetwork as PPOActor, CriticNetwork as PPOCritic
from .ddpg import DDPGAgent, Actor as DDPGActor, Critic as DDPGCritic, OrnsteinUhlenbeckNoise
from .td3 import TD3Agent, Actor as TD3Actor, Critic as TD3Critic
from .sac import SACAgent, GaussianActor, Critic as SACCritic

# Import utility functions
from .ppo import evaluate_agent as evaluate_ppo, plot_training_results as plot_ppo
from .ddpg import evaluate_agent as evaluate_ddpg, plot_training_results as plot_ddpg
from .td3 import evaluate_agent as evaluate_td3, plot_training_results as plot_td3
from .sac import evaluate_agent as evaluate_sac, plot_training_results as plot_sac

# Import replay buffer (shared between off-policy algorithms)
from .ddpg import ReplayBuffer

__all__ = [
    # Agents
    'PPOAgent',
    'DDPGAgent',
    'TD3Agent',
    'SACAgent',

    # Networks
    'PPOActor',
    'PPOCritic',
    'DDPGActor',
    'DDPGCritic',
    'TD3Actor',
    'TD3Critic',
    'GaussianActor',
    'SACCritic',

    # Utilities
    'ReplayBuffer',
    'OrnsteinUhlenbeckNoise',

    # Evaluation functions
    'evaluate_ppo',
    'evaluate_ddpg',
    'evaluate_td3',
    'evaluate_sac',

    # Plotting functions
    'plot_ppo',
    'plot_ddpg',
    'plot_td3',
    'plot_sac',
]

__version__ = '1.0.0'
__author__ = 'MARK-126'
