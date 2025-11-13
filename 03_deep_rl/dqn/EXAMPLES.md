# Ejemplos de Uso - DQN Variants

Guía práctica de cómo usar las implementaciones de Double DQN y Dueling DQN.

## 🚀 Ejecución Rápida

### 1. Entrenar Double DQN en LunarLander

```bash
cd /home/user/Reinforcement-learning-guide/03_deep_rl/dqn
python double_dqn.py
```

**Salida esperada:**
- Entrenamiento de 600 episodios
- Progreso cada 10 episodios
- Gráficos de rewards y losses
- Modelo guardado cada 100 episodios
- Evaluación final sin exploración

**Archivos generados:**
- `double_dqn_lunarlander-v2_training.png` - Gráficos de entrenamiento
- `double_dqn_lunarlander-v2.pth` - Checkpoints periódicos
- `double_dqn_lunarlander-v2_final.pth` - Modelo final

### 2. Entrenar Dueling DQN en LunarLander

```bash
cd /home/user/Reinforcement-learning-guide/03_deep_rl/dqn
python dueling_dqn.py
```

**Salida esperada:**
- Entrenamiento de 600 episodios con Double DQN
- Análisis de Value/Advantage streams
- Gráficos y modelo guardados

**Archivos generados:**
- `dueling_dqn_lunarlander-v2_training.png` - Entrenamiento
- `dueling_dqn_lunarlander-v2_va_analysis.png` - Análisis V/A
- `dueling_dqn_lunarlander-v2_final.pth` - Modelo final

## 📚 Ejemplos de Código

### Ejemplo 1: Double DQN Básico

```python
import gymnasium as gym
from double_dqn import DoubleDQNAgent, train_double_dqn, evaluate_agent

# Crear ambiente
env = gym.make('CartPole-v1')

# Crear agente
agent = DoubleDQNAgent(
    state_dim=4,
    action_dim=2,
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=500,
    buffer_size=10000,
    batch_size=64
)

# Entrenar
rewards, losses = train_double_dqn(env, agent, n_episodes=300)

# Evaluar
avg_reward = evaluate_agent(env, agent, n_episodes=20)
print(f"Reward promedio: {avg_reward:.2f}")

# Guardar
agent.save("my_double_dqn.pth")
```

### Ejemplo 2: Dueling DQN + Double DQN

```python
import gymnasium as gym
from dueling_dqn import DuelingDQNAgent, train_dueling_dqn

# Crear ambiente
env = gym.make('LunarLander-v2')

# Crear agente con Dueling + Double DQN
agent = DuelingDQNAgent(
    state_dim=8,
    action_dim=4,
    learning_rate=5e-4,
    gamma=0.99,
    epsilon_decay=1000,
    buffer_size=50000,
    batch_size=128,
    tau=0.005,  # Soft update
    use_double_dqn=True  # Combinar con Double DQN
)

# Entrenar
rewards, losses = train_dueling_dqn(env, agent, n_episodes=600)

# Analizar estado
state, _ = env.reset()
analysis = agent.analyze_value_advantage(state)
print(f"V(s) = {analysis['value']:.3f}")
print(f"A(s,a) = {analysis['advantage']}")
print(f"Q(s,a) = {analysis['q_values']}")
```

### Ejemplo 3: Comparar Double vs Standard DQN

```python
import gymnasium as gym
from dqn_basic import DQNAgent, train_dqn
from double_dqn import DoubleDQNAgent, train_double_dqn, compare_with_standard_dqn

env = gym.make('CartPole-v1')

# Entrenar DQN estándar
agent_standard = DQNAgent(state_dim=4, action_dim=2)
rewards_standard, _ = train_dqn(env, agent_standard, n_episodes=300)

# Entrenar Double DQN
agent_double = DoubleDQNAgent(state_dim=4, action_dim=2)
rewards_double, _ = train_double_dqn(env, agent_double, n_episodes=300)

# Comparar
compare_with_standard_dqn(rewards_double, rewards_standard)
```

### Ejemplo 4: Cargar y Continuar Entrenamiento

```python
import gymnasium as gym
from double_dqn import DoubleDQNAgent, train_double_dqn

env = gym.make('LunarLander-v2')

# Crear agente
agent = DoubleDQNAgent(state_dim=8, action_dim=4)

# Cargar modelo previamente entrenado
agent.load("double_dqn_lunarlander-v2.pth")

# Continuar entrenamiento desde donde se quedó
rewards, losses = train_double_dqn(
    env, agent,
    n_episodes=100,  # 100 episodios más
    save_path="double_dqn_continued.pth"
)
```

### Ejemplo 5: Evaluación sin Entrenamiento

```python
import gymnasium as gym
from dueling_dqn import DuelingDQNAgent, evaluate_agent

env = gym.make('LunarLander-v2')

# Crear y cargar agente
agent = DuelingDQNAgent(state_dim=8, action_dim=4)
agent.load("dueling_dqn_lunarlander-v2_final.pth")

# Evaluar múltiples veces
avg_reward = evaluate_agent(
    env, agent,
    n_episodes=50,
    render=False,
    analyze=True  # Mostrar análisis de V/A
)

print(f"Rendimiento del modelo: {avg_reward:.2f}")
```

### Ejemplo 6: Configuración Personalizada

```python
import gymnasium as gym
from dueling_dqn import DuelingDQNAgent, train_dueling_dqn

env = gym.make('LunarLander-v2')

# Configuración personalizada para convergencia rápida
agent = DuelingDQNAgent(
    state_dim=8,
    action_dim=4,
    # Hiperparámetros optimizados
    learning_rate=1e-3,      # Mayor LR
    gamma=0.995,             # Mayor descuento
    epsilon_start=0.5,       # Menos exploración inicial
    epsilon_end=0.01,
    epsilon_decay=500,       # Decaimiento más rápido
    buffer_size=100000,      # Buffer más grande
    batch_size=256,          # Batches más grandes
    tau=0.01,                # Soft update más agresivo
    hidden_dim=512,          # Red más grande
    use_double_dqn=True
)

rewards, losses = train_dueling_dqn(env, agent, n_episodes=400)
```

### Ejemplo 7: Solo Dueling (sin Double DQN)

```python
import gymnasium as gym
from dueling_dqn import DuelingDQNAgent, train_dueling_dqn

env = gym.make('CartPole-v1')

# Usar solo arquitectura Dueling sin Double DQN
agent = DuelingDQNAgent(
    state_dim=4,
    action_dim=2,
    use_double_dqn=False  # Desactivar Double DQN
)

rewards, losses = train_dueling_dqn(env, agent, n_episodes=200)
```

### Ejemplo 8: Análisis de Value/Advantage

```python
import gymnasium as gym
import numpy as np
from dueling_dqn import DuelingDQNAgent, visualize_value_advantage

env = gym.make('LunarLander-v2')
agent = DuelingDQNAgent(state_dim=8, action_dim=4)
agent.load("dueling_dqn_lunarlander-v2_final.pth")

# Visualizar distribución de V y A
visualize_value_advantage(
    agent,
    env,
    n_samples=200,
    save_path="va_distribution.png"
)

# Analizar estados específicos
states_to_analyze = []
for _ in range(10):
    state, _ = env.reset()
    states_to_analyze.append(state)

for i, state in enumerate(states_to_analyze):
    analysis = agent.analyze_value_advantage(state)
    print(f"\nEstado {i}:")
    print(f"  V(s) = {analysis['value']:.3f}")
    print(f"  Mejor acción: {np.argmax(analysis['q_values'])}")
    print(f"  Q-values: {analysis['q_values']}")
```

### Ejemplo 9: Hard vs Soft Update

```python
import gymnasium as gym
from double_dqn import DoubleDQNAgent, train_double_dqn

env = gym.make('CartPole-v1')

# Hard update (cada N episodios)
agent_hard = DoubleDQNAgent(
    state_dim=4,
    action_dim=2,
    target_update=10,  # Cada 10 episodios
    tau=None  # Hard update
)

# Soft update (cada step)
agent_soft = DoubleDQNAgent(
    state_dim=4,
    action_dim=2,
    tau=0.001  # Soft update con factor pequeño
)

# Entrenar ambos
rewards_hard, _ = train_double_dqn(env, agent_hard, n_episodes=200)
rewards_soft, _ = train_double_dqn(env, agent_soft, n_episodes=200)

# Comparar estabilidad
import matplotlib.pyplot as plt
plt.plot(rewards_hard, label='Hard Update', alpha=0.6)
plt.plot(rewards_soft, label='Soft Update', alpha=0.6)
plt.legend()
plt.savefig('hard_vs_soft_update.png')
```

### Ejemplo 10: Importar desde el Paquete

```python
# Importar desde el paquete dqn
from dqn import (
    DQNAgent,           # DQN básico
    DoubleDQNAgent,     # Double DQN
    DuelingDQNAgent,    # Dueling DQN
    train_dqn,
    train_double_dqn,
    train_dueling_dqn
)

import gymnasium as gym

# Usar cualquier variante
env = gym.make('LunarLander-v2')

# Opción 1: DQN básico
agent1 = DQNAgent(state_dim=8, action_dim=4)

# Opción 2: Double DQN
agent2 = DoubleDQNAgent(state_dim=8, action_dim=4)

# Opción 3: Dueling + Double DQN (recomendado)
agent3 = DuelingDQNAgent(state_dim=8, action_dim=4, use_double_dqn=True)

# Entrenar con la función correspondiente
rewards, losses = train_dueling_dqn(env, agent3, n_episodes=500)
```

## 🎯 Casos de Uso Recomendados

### Para Ambientes Simples (CartPole, MountainCar)
```python
# Configuración simple y rápida
agent = DoubleDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=1e-3,
    epsilon_decay=500,
    buffer_size=10000,
    batch_size=64,
    tau=None  # Hard update
)
```

### Para Ambientes Complejos (LunarLander, Atari)
```python
# Configuración robusta
agent = DuelingDQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=5e-4,
    epsilon_decay=1000,
    buffer_size=100000,
    batch_size=128,
    tau=0.005,  # Soft update
    hidden_dim=256,
    use_double_dqn=True  # Combinar mejoras
)
```

### Para Investigación/Experimentación
```python
# Probar diferentes configuraciones
configs = {
    'baseline': {'use_double_dqn': False, 'tau': None},
    'double': {'use_double_dqn': True, 'tau': None},
    'soft': {'use_double_dqn': True, 'tau': 0.001},
}

results = {}
for name, config in configs.items():
    agent = DuelingDQNAgent(state_dim=8, action_dim=4, **config)
    rewards, _ = train_dueling_dqn(env, agent, n_episodes=300)
    results[name] = rewards
```

## 💡 Tips Prácticos

### 1. Debugging
```python
# Verificar que epsilon está decayendo
print(f"Epsilon inicial: {agent.epsilon}")
for _ in range(1000):
    agent.steps += 1
    agent.update_epsilon()
print(f"Epsilon después de 1000 steps: {agent.epsilon}")

# Verificar que el buffer se está llenando
print(f"Buffer size: {len(agent.replay_buffer)}")

# Verificar que la red está aprendiendo
loss1 = agent.train_step()
loss2 = agent.train_step()
print(f"Loss inicial: {loss1}, Loss después: {loss2}")
```

### 2. Monitoreo durante Entrenamiento
```python
# Guardar métricas adicionales
import json

training_log = {
    'config': {
        'learning_rate': agent.optimizer.param_groups[0]['lr'],
        'gamma': agent.gamma,
        'buffer_size': len(agent.replay_buffer.buffer)
    },
    'rewards': rewards,
    'losses': losses
}

with open('training_log.json', 'w') as f:
    json.dump(training_log, f)
```

### 3. Early Stopping
```python
# Parar si alcanza objetivo
target_reward = 200
patience = 50

best_avg = -float('inf')
episodes_without_improvement = 0

for episode in range(n_episodes):
    # ... entrenamiento ...

    if episode >= 10:
        avg_reward = np.mean(rewards[-10:])

        if avg_reward > best_avg:
            best_avg = avg_reward
            episodes_without_improvement = 0
            agent.save('best_model.pth')
        else:
            episodes_without_improvement += 1

        if avg_reward >= target_reward:
            print(f"¡Objetivo alcanzado en episodio {episode}!")
            break

        if episodes_without_improvement >= patience:
            print(f"Stopping early - sin mejora por {patience} episodios")
            break
```

## 🔍 Troubleshooting

### Problema: No aprende (reward estancado)
**Soluciones:**
1. Aumentar `buffer_size` y esperar a que se llene
2. Reducir `learning_rate` (probar 5e-4)
3. Aumentar `epsilon_decay` para más exploración
4. Verificar que el ambiente retorna rewards apropiadas

### Problema: Aprendizaje inestable
**Soluciones:**
1. Usar soft update (`tau=0.001` a `0.01`)
2. Reducir `learning_rate`
3. Aumentar `batch_size`
4. Usar gradient clipping (ya implementado)

### Problema: Memoria insuficiente
**Soluciones:**
1. Reducir `buffer_size`
2. Reducir `batch_size`
3. Reducir `hidden_dim`
4. Entrenar en CPU si GPU da OOM

### Problema: Muy lento
**Soluciones:**
1. Reducir `n_episodes`
2. Usar GPU (`agent.device` detecta automáticamente)
3. Aumentar `batch_size` para mejor uso de GPU
4. Reducir `max_steps` por episodio

---

**¿Necesitas más ayuda?** Consulta el README.md para documentación completa.
