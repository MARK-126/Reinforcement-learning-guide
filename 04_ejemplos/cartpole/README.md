# CartPole Example - Q-Learning con Discretización

Este ejemplo muestra cómo entrenar un agente para resolver el problema de CartPole usando Q-Learning con discretización del espacio de estados continuo.

## Problema

CartPole-v1 es un problema clásico de control donde:
- **Objetivo**: Mantener un poste equilibrado en un carrito
- **Estados**: Posición del carrito, velocidad, ángulo del poste, velocidad angular
- **Acciones**: Empujar carrito a izquierda (0) o derecha (1)
- **Recompensa**: +1 por cada timestep que el poste permanece arriba
- **Terminación**: El poste cae (ángulo > 12°) o el carrito sale del área

## Archivos

- `cartpole_qlearning.py`: Implementación con Q-Learning y discretización
- `cartpole_dqn.py`: Implementación con Deep Q-Network (más avanzada)
- `README.md`: Este archivo

## Instalación

```bash
pip install gymnasium numpy matplotlib
```

## Uso

### Q-Learning Discreto

```bash
python cartpole_qlearning.py
```

Este script:
1. Discretiza el espacio de estados continuo en bins
2. Entrena un agente Q-Learning por 500 episodios
3. Evalúa el agente entrenado
4. Guarda gráficos de resultados

### Deep Q-Network

```bash
python cartpole_dqn.py
```

Usa redes neuronales, no requiere discretización.

## Resultados Esperados

- **Q-Learning**: Resuelve en ~300-400 episodios
- **DQN**: Resuelve en ~150-200 episodios
- **Criterio de éxito**: Recompensa promedio > 195 sobre 100 episodios

## Parámetros Ajustables

### Q-Learning
- `n_bins`: Número de bins para discretización (default: 10)
- `alpha`: Learning rate (default: 0.1)
- `gamma`: Discount factor (default: 0.99)
- `epsilon`: Exploration rate (default: 1.0 → 0.01)

### DQN
- `learning_rate`: Learning rate para optimizer (default: 1e-3)
- `buffer_size`: Tamaño del replay buffer (default: 10000)
- `batch_size`: Tamaño del batch para training (default: 64)
- `target_update`: Frecuencia de actualización de target network (default: 10)

## Comparación de Métodos

| Método | Sample Efficiency | Velocidad | Estabilidad | Complejidad |
|--------|------------------|-----------|-------------|-------------|
| Q-Learning | Baja | Rápido | Alta | Baja |
| DQN | Media | Medio | Media | Media |

## Extensiones Posibles

1. **Double DQN**: Reduce sobreestimación de valores
2. **Dueling DQN**: Arquitectura mejorada
3. **Prioritized Replay**: Muestrea transiciones importantes más frecuentemente
4. **N-step returns**: Usa recompensas multi-step
5. **Noisy Networks**: Exploración en el espacio de parámetros

## Referencias

- Sutton & Barto - Reinforcement Learning: An Introduction
- Mnih et al. (2015) - Human-level control through deep reinforcement learning
- Documentación de Gymnasium: https://gymnasium.farama.org/environments/classic_control/cart_pole/
