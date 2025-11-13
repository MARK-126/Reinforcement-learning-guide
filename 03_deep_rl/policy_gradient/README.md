# Policy Gradient Methods

Implementaciones educativas de algoritmos de Policy Gradient para Deep Reinforcement Learning.

## Contenido

Este módulo contiene implementaciones completas y bien documentadas de:

1. **REINFORCE** (`reinforce.py`) - Monte Carlo Policy Gradient
2. **A2C** (`actor_critic.py`) - Advantage Actor-Critic
3. **Módulo de Python** (`__init__.py`) - Exports y documentación

## Características Principales

### ✅ Implementaciones Completas

- **REINFORCE**: Algoritmo básico de policy gradient con retornos de Monte Carlo
  - Con y sin baseline para reducción de varianza
  - Soporte para espacios discretos y continuos
  - Normalización de ventajas
  - Regularización de entropía

- **A2C**: Actor-Critic con estimación de ventaja
  - Arquitecturas separadas para Actor y Critic
  - GAE (Generalized Advantage Estimation)
  - n-step returns
  - TD error para ventajas
  - Actualizaciones más eficientes que REINFORCE

### 🎯 Características Técnicas

- **PyTorch**: Redes neuronales con autograd
- **Type Hints**: Código completamente tipado
- **Documentación en Español**: Docstrings comprensivos
- **Espacios de Acción**: Soporte discreto y continuo
- **Normalización**: Ventajas normalizadas para estabilidad
- **Gradient Clipping**: Prevención de gradientes explosivos
- **Entropía**: Bonus de exploración configurable
- **Historial**: Tracking completo de métricas

### 📊 Ejemplos Incluidos

Cada implementación incluye sección `__main__` con ejemplos completos:

- **CartPole-v1**: Ambiente discreto (REINFORCE y A2C)
- **Pendulum-v1**: Ambiente continuo (A2C)
- Comparaciones entre variantes (con/sin baseline, con/sin GAE)
- Visualizaciones de entrenamiento
- Evaluación de políticas aprendidas

## Instalación

```bash
# Requisitos
pip install torch gymnasium numpy matplotlib
```

## Uso Rápido

### REINFORCE

```python
import gymnasium as gym
from 03_deep_rl.policy_gradient.reinforce import REINFORCEAgent

# Crear ambiente
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Crear agente REINFORCE con baseline
agent = REINFORCEAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    continuous=False,
    learning_rate=3e-4,
    gamma=0.99,
    use_baseline=True,
    baseline_lr=1e-3,
    entropy_coef=0.01,
    normalize_advantages=True,
    hidden_dims=[128, 128]
)

# Entrenar
history = agent.train(
    env=env,
    n_episodes=500,
    max_steps=500,
    print_every=50
)

# Evaluar
from 03_deep_rl.policy_gradient.reinforce import evaluate_agent
mean_reward, std_reward = evaluate_agent(agent, env, n_episodes=100)
print(f"Reward: {mean_reward:.2f} ± {std_reward:.2f}")

# Guardar
agent.save('reinforce_model.pth')
```

### A2C (Actor-Critic)

```python
import gymnasium as gym
from 03_deep_rl.policy_gradient.actor_critic import A2CAgent

# Crear ambiente
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Crear agente A2C con GAE
agent = A2CAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    continuous=False,
    actor_lr=3e-4,
    critic_lr=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    entropy_coef=0.01,
    value_loss_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantages=True,
    n_steps=None,  # None = episodio completo
    use_gae=True,
    hidden_dims=[256, 256]
)

# Entrenar
history = agent.train(
    env=env,
    n_episodes=500,
    max_steps=500,
    print_every=50
)

# Evaluar
from 03_deep_rl.policy_gradient.actor_critic import evaluate_agent
mean_reward, std_reward = evaluate_agent(agent, env, n_episodes=100)
print(f"Reward: {mean_reward:.2f} ± {std_reward:.2f}")

# Guardar
agent.save('a2c_model.pth')
```

### Acciones Continuas (Pendulum)

```python
import gymnasium as gym
from 03_deep_rl.policy_gradient.actor_critic import A2CAgent

# Ambiente continuo
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# A2C para continuo
agent = A2CAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    continuous=True,  # Importante!
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    entropy_coef=0.001,
    n_steps=5,  # n-step updates
    use_gae=True
)

# Entrenar
history = agent.train(env=env, n_episodes=300, max_steps=200)
```

## Ejecutar Ejemplos

```bash
# REINFORCE en CartPole
python3 -m 03_deep_rl.policy_gradient.reinforce

# A2C en CartPole y Pendulum
python3 -m 03_deep_rl.policy_gradient.actor_critic
```

## Estructura del Código

### REINFORCE (`reinforce.py`)

```
PolicyNetwork
├─ Discrete: Categorical distribution
└─ Continuous: Normal distribution (mean, log_std)

ValueNetwork (Baseline)
└─ V(s) estimator

REINFORCEAgent
├─ get_action()
├─ compute_returns()  # Monte Carlo returns
├─ train_episode()
└─ train()

Funciones auxiliares:
├─ plot_training_results()
└─ evaluate_agent()
```

### A2C (`actor_critic.py`)

```
ActorNetwork
├─ Discrete: Categorical distribution
└─ Continuous: Normal distribution

CriticNetwork
└─ V(s) estimator

A2CAgent
├─ get_action()
├─ compute_gae()          # GAE(λ)
├─ compute_n_step_returns()
├─ train_step()
└─ train()

Funciones auxiliares:
├─ plot_training_results()
└─ evaluate_agent()
```

## Teoría

### REINFORCE

**Objetivo**: Maximizar J(θ) = E[R|π_θ]

**Gradiente de política**:
```
∇J(θ) = E[∇log π_θ(a|s) * G_t]
```

donde G_t es el retorno desde el tiempo t.

**Con baseline**:
```
∇J(θ) = E[∇log π_θ(a|s) * (G_t - b(s))]
```

El baseline b(s) = V(s) reduce varianza sin introducir sesgo.

**Ventajas**:
- Simple de implementar
- Convergencia garantizada a mínimo local
- No requiere modelo del ambiente

**Desventajas**:
- Alta varianza
- Ineficiente en muestras (requiere episodios completos)
- Aprendizaje lento

### A2C (Advantage Actor-Critic)

**Actor**: Actualiza política π_θ(a|s)
```
∇J(θ) = E[∇log π_θ(a|s) * A(s,a)]
```

**Critic**: Actualiza value function V_φ(s)
```
L(φ) = E[(R_t - V_φ(s))²]
```

**Ventaja**: A(s,a) = Q(s,a) - V(s)

**GAE (Generalized Advantage Estimation)**:
```
A^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

donde δ_t = r_t + γV(s_{t+1}) - V(s_t) es el TD error.

**Ventajas**:
- Menor varianza que REINFORCE
- Más eficiente en muestras
- Puede usar n-step o episodios completos
- GAE balancea sesgo-varianza

**Desventajas**:
- Más complejo (dos redes)
- Sesgo del value function
- Requiere tuning de hiperparámetros

## Hiperparámetros

### REINFORCE

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Tasa de aprendizaje del policy |
| `gamma` | 0.99 | Factor de descuento |
| `use_baseline` | True | Usar value network como baseline |
| `baseline_lr` | 1e-3 | Tasa de aprendizaje del baseline |
| `entropy_coef` | 0.01 | Coeficiente de bonus de entropía |
| `normalize_advantages` | True | Normalizar ventajas |
| `hidden_dims` | [128, 128] | Dimensiones de capas ocultas |

### A2C

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `actor_lr` | 3e-4 | Tasa de aprendizaje del actor |
| `critic_lr` | 1e-3 | Tasa de aprendizaje del critic |
| `gamma` | 0.99 | Factor de descuento |
| `gae_lambda` | 0.95 | Lambda para GAE (0=TD(0), 1=MC) |
| `entropy_coef` | 0.01 | Coeficiente de bonus de entropía |
| `value_loss_coef` | 0.5 | Peso del loss del value function |
| `max_grad_norm` | 0.5 | Límite para gradient clipping |
| `normalize_advantages` | True | Normalizar ventajas |
| `n_steps` | None | Steps para actualizar (None=episodio) |
| `use_gae` | True | Usar GAE vs n-step simple |
| `hidden_dims` | [256, 256] | Dimensiones de capas ocultas |

## Comparación de Resultados

### CartPole-v1 (500 episodios)

| Algoritmo | Reward Promedio | Std | Notas |
|-----------|----------------|-----|-------|
| REINFORCE sin baseline | ~300 | ±50 | Alta varianza |
| REINFORCE con baseline | ~450 | ±30 | Más estable |
| A2C sin GAE | ~470 | ±20 | Mejor que REINFORCE |
| A2C con GAE | ~490 | ±15 | Mejor rendimiento |

### Pendulum-v1 (300 episodios)

| Algoritmo | Reward Promedio | Notas |
|-----------|----------------|-------|
| A2C (continuo, GAE) | ~-200 | Buen control |

## Visualizaciones

Ambas implementaciones generan gráficos automáticamente:

- **Rewards por episodio** (con moving average)
- **Longitud de episodios**
- **Policy/Actor Loss**
- **Value/Critic Loss** (si aplica)
- **Entropía** de la política
- **Ventajas promedio** (A2C)

Archivos generados:
- `reinforce_training.png`
- `a2c_training.png`
- etc.

## Testing

Para verificar las implementaciones:

```python
# Test syntax
python3 -m py_compile 03_deep_rl/policy_gradient/reinforce.py
python3 -m py_compile 03_deep_rl/policy_gradient/actor_critic.py

# Run examples
python3 03_deep_rl/policy_gradient/reinforce.py
python3 03_deep_rl/policy_gradient/actor_critic.py
```

## Referencias

### Papers

1. **REINFORCE**:
   - Williams, R. J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"

2. **A2C**:
   - Mnih, V., et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning"

3. **GAE**:
   - Schulman, J., et al. (2016). "High-Dimensional Continuous Control Using Generalized Advantage Estimation"

### Recursos Adicionales

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

## Autor

MARK-126

## Licencia

Educational purposes - Reinforcement Learning Guide
