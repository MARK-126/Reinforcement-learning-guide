# Lunar Lander - Ejemplos Completos

Ejemplos de entrenamiento de agentes para el ambiente **LunarLander-v2** de Gymnasium.

## 📋 Descripción del Problema

LunarLander-v2 es un ambiente clásico de control donde un módulo lunar debe aterrizar de forma segura en una plataforma. El agente controla los propulsores para navegar y aterrizar correctamente.

**Espacio de Estados** (8 dimensiones):
- Posición x, y
- Velocidad x, y
- Ángulo
- Velocidad angular
- Contacto pata izquierda (booleano)
- Contacto pata derecha (booleano)

**Espacio de Acciones** (4 acciones discretas):
- 0: No hacer nada
- 1: Propulsor izquierdo
- 2: Propulsor principal (abajo)
- 3: Propulsor derecho

**Recompensas:**
- Aterrizar en la plataforma: +100-140 puntos
- Cada pata tocando el suelo: +10 puntos
- Propulsor principal: -0.3 por frame
- Propulsores laterales: -0.03 por frame
- Crash o salirse: -100 puntos

**Criterio de Éxito:**
Promedio de 200+ puntos en 100 episodios consecutivos.

## 🚀 Ejemplos Disponibles

### 1. `train_dqn.py` - DQN y Variantes

Entrena agentes usando Deep Q-Networks:
- DQN básico
- Double DQN
- Dueling DQN (recomendado)

**Uso:**
```bash
# Entrenar Dueling DQN (mejor opción)
python train_dqn.py --agent dueling_dqn --episodes 600

# Entrenar DQN básico
python train_dqn.py --agent dqn --episodes 600

# Comparar todos los DQN
python train_dqn.py --compare
```

**Hiperparámetros recomendados:**
- Learning rate: 5e-4
- Gamma: 0.99
- Epsilon decay: 0.995
- Batch size: 64
- Buffer size: 100,000
- Target update: cada 10 episodios (o tau=0.001 para soft update)

**Resultados esperados:**
- DQN: ~200-220 puntos en ~500-600 episodios
- Double DQN: ~210-230 puntos en ~450-550 episodios
- Dueling DQN: ~220-250 puntos en ~400-500 episodios

### 2. `train_advanced.py` - PPO

Entrena agentes usando Proximal Policy Optimization.

**Uso:**
```bash
# Entrenar PPO
python train_advanced.py --episodes 300

# Comparar PPO vs DQN
python train_advanced.py --compare
```

**Hiperparámetros recomendados:**
- Actor LR: 3e-4
- Critic LR: 1e-3
- Gamma: 0.99
- GAE lambda: 0.95
- Clip epsilon: 0.2
- Entropy coef: 0.01
- Epochs: 10
- Batch size: 64

**Resultados esperados:**
- PPO: ~220-260 puntos en ~250-350 episodios

**Ventajas de PPO:**
- Más sample-efficient
- Convergencia más estable
- Menos hiperparámetros sensibles
- Mejor para ambientes con estados continuos

## 📊 Comparación de Algoritmos

| Algoritmo | Episodios hasta éxito | Puntuación final | Estabilidad | Sample Efficiency |
|-----------|----------------------|------------------|-------------|-------------------|
| DQN | ~500-600 | 200-220 | Media | Media |
| Double DQN | ~450-550 | 210-230 | Media-Alta | Media |
| Dueling DQN | ~400-500 | 220-250 | Alta | Media-Alta |
| PPO | ~250-350 | 220-260 | Muy Alta | Alta |

## 💡 Tips para Mejor Rendimiento

### 1. Reward Shaping (opcional)
Puedes modificar las recompensas para acelerar el aprendizaje:
```python
# Penalizar más por usar propulsores
reward = reward - 0.1 * abs(action - 0)  # action 0 = no hacer nada

# Recompensar por reducir velocidad
reward += 0.1 * (1.0 - abs(velocity))
```

### 2. Normalización de Estados
```python
# Normalizar posiciones y velocidades
state = (state - mean) / std
```

### 3. Early Stopping
Detener cuando se alcance el objetivo:
```python
if np.mean(rewards[-100:]) >= 200:
    break
```

### 4. Guardar Mejores Modelos
```python
if total_reward > best_reward:
    agent.save('best_model.pth')
    best_reward = total_reward
```

## 🎮 Visualizar Agente Entrenado

```python
import gymnasium as gym

# Cargar agente
agent.load('lunar_lander_ppo.pth')

# Crear ambiente con render
env = gym.make('LunarLander-v2', render_mode='human')

# Ejecutar episodio
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.get_action(state, training=False)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Recompensa total: {total_reward:.2f}")
env.close()
```

## 📈 Curvas de Aprendizaje Típicas

**Fase 1: Exploración caótica** (0-50 episodios)
- Recompensas: -300 a -100
- Agente explora aleatoriamente
- Alta varianza

**Fase 2: Aprendiendo a no crashear** (50-150 episodios)
- Recompensas: -100 a +50
- Agente aprende a usar propulsores
- Aterrizajes ocasionales

**Fase 3: Refinamiento** (150-300 episodios)
- Recompensas: +50 a +150
- Aterrizajes más frecuentes
- Optimización de fuel

**Fase 4: Maestría** (300+ episodios)
- Recompensas: +150 a +250
- Aterrizajes consistentes
- Uso eficiente de combustible

## 🐛 Troubleshooting

### Problema: El agente no aprende
**Soluciones:**
- Reducir learning rate (probar 1e-4, 5e-5)
- Aumentar epsilon_decay (más exploración)
- Verificar normalización de estados
- Revisar reward clipping

### Problema: Inestabilidad en el entrenamiento
**Soluciones:**
- Usar gradient clipping (max_grad_norm=0.5)
- Reducir learning rate
- Aumentar batch size
- Usar soft target updates (tau=0.001)

### Problema: Overfitting a una estrategia subóptima
**Soluciones:**
- Aumentar entropy coefficient (más exploración)
- Probar PPO en lugar de DQN
- Reward shaping para guiar exploración
- Exploring starts

## 📚 Referencias

- **Paper LunarLander**: [Gymnasium Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- **DQN**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- **Double DQN**: van Hasselt et al. (2016) - "Deep Reinforcement Learning with Double Q-learning"
- **Dueling DQN**: Wang et al. (2016) - "Dueling Network Architectures for Deep Reinforcement Learning"
- **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"

## 🎯 Próximos Pasos

Después de dominar LunarLander:
1. Probar con **BipedalWalker** (control continuo más complejo)
2. Intentar **LunarLanderContinuous-v2** (acciones continuas)
3. Explorar **Atari games** con DQN
4. Implementar **curriculum learning** (empezar con tareas más fáciles)

---

**¡Buena suerte entrenando tu agente lunar! 🚀🌙**
