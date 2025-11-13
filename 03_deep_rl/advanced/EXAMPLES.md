# Ejemplos Prácticos - Algoritmos Avanzados

Esta guía contiene ejemplos prácticos y casos de uso para cada algoritmo implementado.

## 📋 Índice

1. [PPO - Proximal Policy Optimization](#1-ppo---proximal-policy-optimization)
2. [DDPG - Deep Deterministic Policy Gradient](#2-ddpg---deep-deterministic-policy-gradient)
3. [TD3 - Twin Delayed DDPG](#3-td3---twin-delayed-ddpg)
4. [SAC - Soft Actor-Critic](#4-sac---soft-actor-critic)
5. [Comparación Side-by-Side](#5-comparación-side-by-side)
6. [Casos de Uso Avanzados](#6-casos-de-uso-avanzados)

---

## 1. PPO - Proximal Policy Optimization

### Ejemplo 1: CartPole (Acciones Discretas)

```python
import gymnasium as gym
from advanced.ppo import PPOAgent, evaluate_agent, plot_training_results

# Crear ambiente
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]  # 4
action_dim = env.action_space.n  # 2

# Crear agente PPO
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    continuous=False,           # Acciones discretas
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    epsilon_clip=0.2,           # Clipping ratio
    value_clip=0.2,             # Value clipping
    entropy_coef=0.01,          # Entropy bonus
    n_epochs=10,                # Epochs de optimización
    batch_size=64,
    hidden_dims=[64, 64]
)

# Entrenar
print("Entrenando PPO en CartPole-v1...")
history = agent.train(
    env=env,
    n_episodes=300,
    max_steps=500,
    update_interval=2048,       # Update cada 2048 steps
    print_every=10,
    save_every=100,
    save_path='ppo_cartpole.pth'
)

# Visualizar resultados
plot_training_results(history, 'ppo_cartpole_training.png')

# Evaluar
mean_reward, std_reward = evaluate_agent(agent, env, n_episodes=100)
print(f"Recompensa promedio: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()
```

**Resultado esperado:** ~500 reward (resuelve CartPole)

### Ejemplo 2: LunarLander (Acciones Discretas)

```python
import gymnasium as gym
from advanced.ppo import PPOAgent

env = gym.make('LunarLander-v2')

agent = PPOAgent(
    state_dim=8,
    action_dim=4,
    continuous=False,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    epsilon_clip=0.2,
    value_clip=0.2,
    entropy_coef=0.01,
    n_epochs=4,                 # Menos epochs para ambiente más complejo
    batch_size=64,
    hidden_dims=[64, 64]
)

history = agent.train(
    env=env,
    n_episodes=1000,
    max_steps=1000,
    update_interval=2048,
    print_every=20
)

env.close()
```

**Resultado esperado:** ~200+ reward después de 1000 episodios

### Ejemplo 3: Pendulum (Acciones Continuas)

```python
import gymnasium as gym
from advanced.ppo import PPOAgent

env = gym.make('Pendulum-v1')

agent = PPOAgent(
    state_dim=3,
    action_dim=1,
    continuous=True,            # ¡Acciones continuas!
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    epsilon_clip=0.2,
    value_clip=None,            # Sin value clipping para continuo
    entropy_coef=0.0,           # Sin entropy bonus para continuo
    n_epochs=10,
    batch_size=64,
    hidden_dims=[64, 64]
)

history = agent.train(
    env=env,
    n_episodes=300,
    max_steps=200,
    update_interval=2048,
    print_every=10
)

env.close()
```

---

## 2. DDPG - Deep Deterministic Policy Gradient

### Ejemplo 1: Pendulum

```python
import gymnasium as gym
from advanced.ddpg import DDPGAgent, evaluate_agent, plot_training_results

# Crear ambiente
env = gym.make('Pendulum-v1')

# Crear agente DDPG
agent = DDPGAgent(
    state_dim=3,
    action_dim=1,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.001,                  # Soft update factor
    buffer_size=100000,
    batch_size=64,
    noise_type='ou',            # Ornstein-Uhlenbeck noise
    noise_std=0.2,
    hidden_dims=[400, 300]
)

# Entrenar
history = agent.train(
    env=env,
    n_episodes=200,
    max_steps=200,
    warmup_steps=1000,          # Exploración aleatoria inicial
    noise_decay=0.9995,         # Decaimiento del ruido
    min_noise=0.1,
    print_every=10,
    save_path='ddpg_pendulum.pth'
)

plot_training_results(history, 'ddpg_pendulum_training.png')

# Evaluar sin ruido
mean_reward, std_reward = evaluate_agent(agent, env, n_episodes=50)
print(f"Recompensa: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()
```

### Ejemplo 2: MountainCarContinuous con Gaussian Noise

```python
import gymnasium as gym
from advanced.ddpg import DDPGAgent

env = gym.make('MountainCarContinuous-v0')

agent = DDPGAgent(
    state_dim=2,
    action_dim=1,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.001,
    buffer_size=50000,
    batch_size=64,
    noise_type='gaussian',      # Gaussian noise (más simple)
    noise_std=0.3,
    hidden_dims=[400, 300]
)

history = agent.train(
    env=env,
    n_episodes=500,
    max_steps=999,
    warmup_steps=2000,
    noise_decay=0.999,
    min_noise=0.05,
    print_every=20
)

env.close()
```

### Ejemplo 3: Cargar y Continuar Entrenamiento

```python
import gymnasium as gym
from advanced.ddpg import DDPGAgent

env = gym.make('Pendulum-v1')

# Crear agente
agent = DDPGAgent(state_dim=3, action_dim=1)

# Cargar checkpoint
agent.load('ddpg_pendulum.pth')

# Continuar entrenamiento
history = agent.train(
    env=env,
    n_episodes=100,             # 100 episodios adicionales
    max_steps=200,
    warmup_steps=0,             # Sin warmup (ya entrenado)
    noise_decay=0.999,
    min_noise=0.05
)

env.close()
```

---

## 3. TD3 - Twin Delayed DDPG

### Ejemplo 1: Pendulum

```python
import gymnasium as gym
from advanced.td3 import TD3Agent, evaluate_agent, plot_training_results

# Crear ambiente
env = gym.make('Pendulum-v1')
max_action = float(env.action_space.high[0])  # 2.0

# Crear agente TD3
agent = TD3Agent(
    state_dim=3,
    action_dim=1,
    max_action=max_action,      # Límite de acción
    actor_lr=3e-4,
    critic_lr=3e-4,
    gamma=0.99,
    tau=0.005,                  # Soft update
    policy_noise=0.2,           # Target policy smoothing
    noise_clip=0.5,             # Límite del ruido
    policy_delay=2,             # Update actor cada 2 steps
    buffer_size=1000000,
    batch_size=256,
    exploration_noise=0.1,      # Ruido de exploración
    hidden_dims=[256, 256]
)

# Entrenar
history = agent.train(
    env=env,
    n_episodes=200,
    max_steps=200,
    warmup_steps=1000,
    noise_decay=0.999,
    min_noise=0.1,
    print_every=10,
    save_path='td3_pendulum.pth'
)

plot_training_results(history, 'td3_pendulum_training.png')

mean_reward, std_reward = evaluate_agent(agent, env, n_episodes=50)
print(f"Recompensa: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()
```

### Ejemplo 2: MountainCarContinuous

```python
import gymnasium as gym
from advanced.td3 import TD3Agent

env = gym.make('MountainCarContinuous-v0')
max_action = float(env.action_space.high[0])

agent = TD3Agent(
    state_dim=2,
    action_dim=1,
    max_action=max_action,
    actor_lr=3e-4,
    critic_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
    buffer_size=1000000,
    batch_size=256,
    exploration_noise=0.1,
    hidden_dims=[256, 256]
)

history = agent.train(
    env=env,
    n_episodes=500,
    max_steps=999,
    warmup_steps=5000,          # Más warmup para ambiente difícil
    noise_decay=0.995,
    min_noise=0.05,
    print_every=20
)

env.close()
```

### Ejemplo 3: Análisis de Twin Critics

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from advanced.td3 import TD3Agent

env = gym.make('Pendulum-v1')
agent = TD3Agent(state_dim=3, action_dim=1, max_action=2.0)

# Entrenar
history = agent.train(env, n_episodes=200)

# Analizar diferencia entre Q1 y Q2
q1_values = np.array(history['q1_values'])
q2_values = np.array(history['q2_values'])
q_diff = q1_values - q2_values

plt.figure(figsize=(10, 4))
plt.plot(q_diff, alpha=0.6, label='Q1 - Q2')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Q-value Difference')
plt.title('Twin Q-networks Divergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('td3_twin_analysis.png')
plt.close()

print(f"Q-diff mean: {q_diff.mean():.3f}")
print(f"Q-diff std: {q_diff.std():.3f}")

env.close()
```

---

## 4. SAC - Soft Actor-Critic

### Ejemplo 1: Pendulum con Auto-tuning

```python
import gymnasium as gym
from advanced.sac import SACAgent, evaluate_agent, plot_training_results

# Crear ambiente
env = gym.make('Pendulum-v1')

# Crear agente SAC con auto-tuning
agent = SACAgent(
    state_dim=3,
    action_dim=1,
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,              # Learning rate para α
    gamma=0.99,
    tau=0.005,
    alpha=0.2,                  # Valor inicial (si no auto-tune)
    auto_tune=True,             # ¡Auto-tuning activado!
    target_entropy=None,        # Usa -action_dim
    buffer_size=1000000,
    batch_size=256,
    hidden_dims=[256, 256]
)

# Entrenar
history = agent.train(
    env=env,
    n_episodes=150,
    max_steps=200,
    warmup_steps=1000,
    updates_per_step=1,         # Updates por step del ambiente
    print_every=10,
    save_path='sac_pendulum.pth'
)

plot_training_results(history, 'sac_pendulum_training.png')

# Evaluar (deterministic)
mean_reward, std_reward = evaluate_agent(agent, env, n_episodes=50)
print(f"Recompensa: {mean_reward:.2f} ± {std_reward:.2f}")

# Verificar α final
print(f"Temperature α final: {agent.alpha:.3f}")

env.close()
```

**¿Por qué auto-tune?** SAC ajusta automáticamente el balance entre exploración (entropía) y explotación (reward).

### Ejemplo 2: MountainCarContinuous

```python
import gymnasium as gym
from advanced.sac import SACAgent

env = gym.make('MountainCarContinuous-v0')

agent = SACAgent(
    state_dim=2,
    action_dim=1,
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    auto_tune=True,
    buffer_size=1000000,
    batch_size=256,
    hidden_dims=[256, 256]
)

history = agent.train(
    env=env,
    n_episodes=400,
    max_steps=999,
    warmup_steps=5000,
    updates_per_step=1,
    print_every=20
)

env.close()
```

### Ejemplo 3: SAC sin Auto-tuning (α fijo)

```python
import gymnasium as gym
from advanced.sac import SACAgent

env = gym.make('Pendulum-v1')

agent = SACAgent(
    state_dim=3,
    action_dim=1,
    alpha=0.1,                  # α fijo
    auto_tune=False,            # Sin auto-tuning
    buffer_size=1000000,
    batch_size=256
)

history = agent.train(
    env=env,
    n_episodes=200,
    warmup_steps=1000
)

env.close()
```

**Nota:** Auto-tuning es generalmente mejor. Úsalo a menos que tengas una razón específica.

### Ejemplo 4: Análisis de Entropía

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from advanced.sac import SACAgent

env = gym.make('Pendulum-v1')
agent = SACAgent(state_dim=3, action_dim=1, auto_tune=True)

# Entrenar
history = agent.train(env, n_episodes=150)

# Analizar evolución de α y entropía
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# α (temperature)
ax1.plot(history['alphas'], alpha=0.6, label='α')
ax1.set_xlabel('Episode')
ax1.set_ylabel('α')
ax1.set_title('Temperature Evolution (Auto-tuned)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Entropía
ax2.plot(history['entropies'], alpha=0.6, label='Entropy')
ax2.axhline(y=-agent.target_entropy, color='r', linestyle='--',
           label=f'Target: {-agent.target_entropy:.2f}')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Entropy')
ax2.set_title('Policy Entropy Evolution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sac_entropy_analysis.png')
plt.close()

env.close()
```

---

## 5. Comparación Side-by-Side

### Mismo Ambiente, Todos los Algoritmos

```python
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from advanced import PPOAgent, DDPGAgent, TD3Agent, SACAgent

# Ambiente común
env_name = 'Pendulum-v1'
n_episodes = 200

# Resultados para almacenar
results = {}

# 1. PPO
print("=" * 50)
print("Entrenando PPO...")
env = gym.make(env_name)
ppo = PPOAgent(state_dim=3, action_dim=1, continuous=True)
ppo_history = ppo.train(env, n_episodes=n_episodes, print_every=50)
results['PPO'] = ppo_history['episode_rewards']
env.close()

# 2. DDPG
print("=" * 50)
print("Entrenando DDPG...")
env = gym.make(env_name)
ddpg = DDPGAgent(state_dim=3, action_dim=1)
ddpg_history = ddpg.train(env, n_episodes=n_episodes, print_every=50)
results['DDPG'] = ddpg_history['episode_rewards']
env.close()

# 3. TD3
print("=" * 50)
print("Entrenando TD3...")
env = gym.make(env_name)
td3 = TD3Agent(state_dim=3, action_dim=1, max_action=2.0)
td3_history = td3.train(env, n_episodes=n_episodes, print_every=50)
results['TD3'] = td3_history['episode_rewards']
env.close()

# 4. SAC
print("=" * 50)
print("Entrenando SAC...")
env = gym.make(env_name)
sac = SACAgent(state_dim=3, action_dim=1, auto_tune=True)
sac_history = sac.train(env, n_episodes=n_episodes, print_every=50)
results['SAC'] = sac_history['episode_rewards']
env.close()

# Comparar resultados
plt.figure(figsize=(14, 6))

for algo_name, rewards in results.items():
    # Moving average
    window = 20
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), moving_avg,
            label=algo_name, linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title(f'Comparación de Algoritmos en {env_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('algorithms_comparison.png', dpi=150)
plt.close()

# Estadísticas finales (últimos 50 episodios)
print("\n" + "=" * 50)
print("RESULTADOS FINALES (últimos 50 episodios)")
print("=" * 50)
for algo_name, rewards in results.items():
    final_rewards = rewards[-50:]
    print(f"{algo_name:8s}: {np.mean(final_rewards):6.2f} ± {np.std(final_rewards):5.2f}")
```

**Resultado esperado en Pendulum:**
```
PPO     : -200.45 ± 45.32
DDPG    : -195.23 ± 38.67
TD3     : -152.34 ± 28.91
SAC     : -148.76 ± 25.13
```

---

## 6. Casos de Uso Avanzados

### Caso 1: Transfer Learning

Entrenar en un ambiente, afinar en otro:

```python
import gymnasium as gym
from advanced.sac import SACAgent

# 1. Pre-entrenar en Pendulum
env1 = gym.make('Pendulum-v1')
agent = SACAgent(state_dim=3, action_dim=1, auto_tune=True)
agent.train(env1, n_episodes=100)
agent.save('pretrained_pendulum.pth')
env1.close()

# 2. Fine-tune en variante más difícil (con ruido)
class NoisyPendulum(gym.Wrapper):
    def step(self, action):
        noise = np.random.randn(*action.shape) * 0.1
        return super().step(action + noise)

env2 = NoisyPendulum(gym.make('Pendulum-v1'))
agent.load('pretrained_pendulum.pth')

# Entrenar con learning rate reducido
agent.actor_optimizer.param_groups[0]['lr'] = 1e-5
agent.critic_optimizer.param_groups[0]['lr'] = 1e-5

agent.train(env2, n_episodes=50, warmup_steps=0)
env2.close()
```

### Caso 2: Curriculum Learning

Aumentar dificultad progresivamente:

```python
import gymnasium as gym
from advanced.ppo import PPOAgent

# Crear agente
agent = PPOAgent(state_dim=4, action_dim=2, continuous=False)

# Curriculum: aumentar longitud máxima del episodio
max_steps_schedule = [200, 300, 400, 500]

for i, max_steps in enumerate(max_steps_schedule):
    print(f"\nCurriculum Level {i+1}: max_steps={max_steps}")

    env = gym.make('CartPole-v1')
    agent.train(
        env,
        n_episodes=100,
        max_steps=max_steps,
        print_every=25
    )
    env.close()

    agent.save(f'curriculum_level_{i+1}.pth')
```

### Caso 3: Ensemble de Políticas

Combinar múltiples agentes:

```python
import gymnasium as gym
import numpy as np
from advanced.sac import SACAgent

# Entrenar 5 agentes independientes
env = gym.make('Pendulum-v1')
agents = []

for i in range(5):
    print(f"\nEntrenando agente {i+1}/5...")
    agent = SACAgent(state_dim=3, action_dim=1, auto_tune=True)
    agent.train(env, n_episodes=100, print_every=50)
    agents.append(agent)

# Evaluar ensemble
def ensemble_action(state, agents):
    """Promedio de acciones de todos los agentes"""
    actions = [agent.get_action(state, deterministic=True)
              for agent in agents]
    return np.mean(actions, axis=0)

# Test ensemble
state, _ = env.reset()
episode_reward = 0

for _ in range(200):
    action = ensemble_action(state, agents)
    state, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward
    if terminated or truncated:
        break

print(f"\nEnsemble reward: {episode_reward:.2f}")
env.close()
```

### Caso 4: Experimento con Sweep de Hyperparameters

```python
import gymnasium as gym
import itertools
import json
from advanced.sac import SACAgent

# Grid search
param_grid = {
    'actor_lr': [1e-4, 3e-4, 1e-3],
    'batch_size': [128, 256],
    'hidden_dims': [[128, 128], [256, 256]]
}

# Generar combinaciones
keys = param_grid.keys()
values = param_grid.values()
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []

for i, params in enumerate(combinations):
    print(f"\nExperimento {i+1}/{len(combinations)}")
    print(f"Params: {params}")

    env = gym.make('Pendulum-v1')

    agent = SACAgent(
        state_dim=3,
        action_dim=1,
        actor_lr=params['actor_lr'],
        batch_size=params['batch_size'],
        hidden_dims=params['hidden_dims'],
        auto_tune=True
    )

    history = agent.train(env, n_episodes=100, print_every=50)
    final_reward = np.mean(history['episode_rewards'][-20:])

    results.append({
        'params': params,
        'final_reward': final_reward
    })

    env.close()

# Encontrar mejores parámetros
best = max(results, key=lambda x: x['final_reward'])
print("\n" + "=" * 50)
print("MEJORES PARÁMETROS:")
print(json.dumps(best, indent=2))

# Guardar resultados
with open('hyperparameter_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Caso 5: Monitoreo con TensorBoard

```python
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from advanced.sac import SACAgent

# Crear writer de TensorBoard
writer = SummaryWriter('runs/sac_experiment')

env = gym.make('Pendulum-v1')
agent = SACAgent(state_dim=3, action_dim=1, auto_tune=True)

# Entrenar con logging a TensorBoard
for episode in range(200):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(200):
        action = agent.get_action(state, deterministic=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)

        if len(agent.replay_buffer) > agent.batch_size:
            metrics = agent.train_step()
            if metrics:
                # Log a TensorBoard
                writer.add_scalar('Loss/Actor', metrics['actor_loss'], episode)
                writer.add_scalar('Loss/Critic', metrics['critic_loss'], episode)
                writer.add_scalar('Value/Q1', metrics['mean_q1'], episode)
                writer.add_scalar('Value/Q2', metrics['mean_q2'], episode)
                writer.add_scalar('Policy/Alpha', metrics['alpha'], episode)
                writer.add_scalar('Policy/Entropy', metrics['entropy'], episode)

        episode_reward += reward
        state = next_state

        if done:
            break

    writer.add_scalar('Episode/Reward', episode_reward, episode)
    print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

writer.close()
env.close()

# Ver resultados: tensorboard --logdir=runs
```

---

## 📊 Resumen de Ejemplos

### Complejidad por Ejemplo

| Ejemplo | Algoritmo | Dificultad | Tiempo Aprox. |
|---------|-----------|------------|---------------|
| CartPole | PPO | ⭐ Fácil | ~5 min |
| Pendulum | DDPG/TD3/SAC | ⭐⭐ Medio | ~10 min |
| LunarLander | PPO | ⭐⭐⭐ Medio | ~30 min |
| MountainCar | TD3/SAC | ⭐⭐⭐⭐ Difícil | ~60 min |
| Comparación | Todos | ⭐⭐⭐ Medio | ~40 min |
| Transfer Learning | SAC | ⭐⭐⭐⭐ Avanzado | Variable |
| Hyperparameter Sweep | SAC | ⭐⭐⭐⭐⭐ Avanzado | Horas |

### Tips por Ambiente

**CartPole-v1:**
- Fácil, resuelve rápido
- Usa PPO con epsilon_clip=0.2
- 300 episodios suficientes

**LunarLander-v2:**
- Medio, requiere exploración
- PPO con entropy_coef=0.01
- 1000 episodios para buenos resultados

**Pendulum-v1:**
- Medio, buen benchmark
- SAC converge más rápido que DDPG/TD3
- 150-200 episodios

**MountainCarContinuous-v0:**
- Difícil, sparse rewards
- Necesita warmup largo (5000 steps)
- TD3/SAC funcionan mejor
- 400-500 episodios

---

## 🎯 Siguiente Paso

1. **Empieza con PPO en CartPole** - más fácil
2. **Prueba SAC en Pendulum** - aprende off-policy
3. **Compara algoritmos** - entiende diferencias
4. **Experimenta con hyperparameters** - tunea para tu problema
5. **Implementa en tu ambiente** - aplica lo aprendido

---

**Autor:** MARK-126
**Última actualización:** 2025
