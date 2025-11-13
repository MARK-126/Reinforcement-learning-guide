"""
Deep Q-Networks (DQN)
=====================

Implementaciones de DQN y sus variantes:
- DQN básico: Q-learning con redes neuronales profundas
- Double DQN: Reduce sobreestimación de Q-values
- Dueling DQN: Separa valor de estado y ventajas de acciones

Cada variante incluye:
- Arquitectura de red neuronal
- Experience replay buffer
- Exploración ε-greedy con decaimiento
- Target network con actualizaciones hard/soft
- Funciones de entrenamiento y evaluación
- Visualización de resultados
- Capacidad de guardar/cargar modelos
"""

# DQN Básico
from .dqn_basic import (
    DQN,
    DQNAgent,
    ReplayBuffer,
    train_dqn,
    plot_training
)

# Double DQN
from .double_dqn import (
    DoubleDQNAgent,
    train_double_dqn,
    evaluate_agent as evaluate_double_dqn,
    compare_with_standard_dqn
)

# Dueling DQN
from .dueling_dqn import (
    DuelingDQN,
    DuelingDQNAgent,
    train_dueling_dqn,
    evaluate_agent as evaluate_dueling_dqn,
    visualize_value_advantage
)

__all__ = [
    # DQN Básico
    'DQN',
    'DQNAgent',
    'ReplayBuffer',
    'train_dqn',
    'plot_training',
    # Double DQN
    'DoubleDQNAgent',
    'train_double_dqn',
    'evaluate_double_dqn',
    'compare_with_standard_dqn',
    # Dueling DQN
    'DuelingDQN',
    'DuelingDQNAgent',
    'train_dueling_dqn',
    'evaluate_dueling_dqn',
    'visualize_value_advantage',
]
