# Algoritmos Avanzados de Deep RL

Este directorio contiene implementaciones de algoritmos avanzados de Deep Reinforcement Learning que representan el **estado del arte** en control continuo y aprendizaje por refuerzo.

## 📁 Archivos Implementados

### 1. `ppo.py` - Proximal Policy Optimization ⭐ RECOMENDADO

Implementación de PPO (Schulman et al., 2017), el algoritmo on-policy más popular y confiable.

**¿Por qué PPO?**
- Simple y robusto - fácil de implementar y tunear
- Sample efficient - reutiliza datos con múltiples epochs
- Stable - clipping previene updates demasiado grandes
- Versatile - funciona bien en diversos ambientes
- SOTA - usado en OpenAI, DeepMind, etc.

**Características clave:**
- **Clipped Surrogate Objective**: Limita cambios de política
  ```
  L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
  ```
- **GAE (Generalized Advantage Estimation)**: Reduce varianza
- **Mini-batch Training**: Múltiples epochs sobre datos recolectados
- **Value Function Clipping**: Opcional, mejora estabilidad
- **Entropy Bonus**: Fomenta exploración
- **Soporte Discrete/Continuous**: Ambos tipos de acciones

**Componentes:**
- `ActorNetwork`: Política π(a|s)
- `CriticNetwork`: Función de valor V(s)
- `PPOAgent`: Agente completo con clipping
- Funciones: `train()`, `evaluate_agent()`, `plot_training_results()`

**Hiperparámetros importantes:**
- `epsilon_clip`: 0.2 (ratio clipping)
- `value_clip`: 0.2 o None (value function clipping)
- `gae_lambda`: 0.95 (GAE parámetro)
- `n_epochs`: 10 (epochs de optimización)
- `batch_size`: 64 (tamaño de mini-batch)
- `update_interval`: 2048 (steps antes de actualizar)

**Cuándo usar PPO:**
- Primera opción para la mayoría de problemas
- Robótica, juegos, problemas diversos
- Cuando necesitas algo confiable y estable
- Para aprendizaje con datos limitados

---

### 2. `ddpg.py` - Deep Deterministic Policy Gradient

Implementación de DDPG (Lillicrap et al., 2016), un actor-critic off-policy para control continuo.

**Concepto:** Extiende DQN a espacios de acción continuos usando una política determinista.

**Características clave:**
- **Deterministic Policy**: μ(s) mapea estados a acciones
- **Actor-Critic**: Combina policy gradient con Q-learning
- **Experience Replay**: Buffer para sample efficiency
- **Target Networks**: Para actor y critic (estabilidad)
- **Ornstein-Uhlenbeck Noise**: Exploración temporalmente correlacionada
- **Gaussian Noise**: Alternativa más simple

**Arquitectura:**
```
Actor:  s → μ_θ(s) → a
Critic: (s,a) → Q_φ(s,a) → Q-value
```

**Componentes:**
- `Actor`: Política determinista
- `Critic`: Q-function Q(s,a)
- `OrnsteinUhlenbeckNoise`: Ruido para exploración
- `ReplayBuffer`: Buffer de experiencias
- `DDPGAgent`: Agente completo
- Funciones: `train()`, `evaluate_agent()`, `plot_training_results()`

**Hiperparámetros importantes:**
- `actor_lr`: 1e-4
- `critic_lr`: 1e-3
- `tau`: 0.001 (soft update)
- `buffer_size`: 100000
- `batch_size`: 64
- `noise_type`: 'ou' o 'gaussian'
- `noise_std`: 0.2

**Cuándo usar DDPG:**
- Control continuo básico
- Prototipado rápido
- Base para aprender algoritmos más avanzados
- Cuando TD3/SAC son overkill

**Limitaciones:**
- Sensible a hiperparámetros
- Puede ser inestable
- Superado por TD3 y SAC

---

### 3. `td3.py` - Twin Delayed DDPG ⭐ RECOMENDADO

Implementación de TD3 (Fujimoto et al., 2018), una mejora significativa sobre DDPG.

**Tres innovaciones clave sobre DDPG:**

1. **Twin Q-networks (Clipped Double Q-learning)**:
   ```python
   Q_target = min(Q1_target, Q2_target)
   ```
   Reduce sobreestimación de Q-values

2. **Delayed Policy Updates**:
   ```python
   if step % policy_delay == 0:
       update_actor()
   ```
   Actualiza actor menos frecuentemente que critic

3. **Target Policy Smoothing**:
   ```python
   noise = clip(N(0, σ), -c, c)
   a' = clip(μ(s') + noise, -max, max)
   ```
   Suaviza superficie de Q, más robusto

**Características:**
- Todas las de DDPG
- Twin critics para doble robustez
- Policy updates retardados
- Smoothing noise en targets
- Más estable que DDPG
- Mismo tipo de acciones (continuous)

**Componentes:**
- `Actor`: Política determinista
- `Critic`: Twin Q-networks (Q1, Q2)
- `ReplayBuffer`: Buffer compartido
- `TD3Agent`: Agente con tres mejoras
- Funciones: `train()`, `evaluate_agent()`, `plot_training_results()`

**Hiperparámetros importantes:**
- `actor_lr`: 3e-4
- `critic_lr`: 3e-4
- `tau`: 0.005 (soft update)
- `policy_noise`: 0.2 (target smoothing)
- `noise_clip`: 0.5 (límite de ruido)
- `policy_delay`: 2 (delayed updates)
- `batch_size`: 256

**Cuándo usar TD3:**
- Control continuo con mejor estabilidad que DDPG
- Cuando DDPG es inestable
- Problemas que requieren determinismo
- Alternativa a SAC más simple

**Ventajas sobre DDPG:**
- Mucho más estable
- Menor sobreestimación
- Mejor rendimiento general
- Mínimo overhead computacional

---

### 4. `sac.py` - Soft Actor-Critic ⭐⭐ ESTADO DEL ARTE

Implementación de SAC (Haarnoja et al., 2018-2019), el estado del arte para control continuo.

**Concepto:** Maximum Entropy RL - maximiza recompensas **y** entropía de la política.

```
J(π) = E[Σ r_t + α H(π(·|s_t))]
```

**Cinco características clave:**

1. **Maximum Entropy Framework**:
   - Fomenta exploración naturalmente
   - Aprende políticas robustas y multimodales
   - Balance entre exploración y explotación

2. **Automatic Temperature Tuning**:
   - α se ajusta automáticamente
   - No requiere tuning manual
   - Mantiene entropía objetivo

3. **Twin Q-networks**:
   - Como TD3, reduce sobreestimación
   - Dos critics independientes

4. **Stochastic Policy**:
   - Gaussian policy: π(a|s) = N(μ(s), σ(s))
   - Reparameterization trick para gradientes
   - Más robusta que deterministic

5. **Off-policy**:
   - Sample efficient con replay buffer
   - Reutiliza experiencias pasadas

**Arquitectura:**
```
Actor:  s → π_θ(a|s) ~ N(μ, σ) → a (stochastic)
Critic: (s,a) → Q_φ(s,a) → Q-value (twin)
Alpha:  α adaptativo (learnable parameter)
```

**Componentes:**
- `GaussianActor`: Política estocástica con reparameterization
- `Critic`: Twin Q-networks
- `ReplayBuffer`: Buffer de experiencias
- `SACAgent`: Agente completo con auto-tuning
- Funciones: `train()`, `evaluate_agent()`, `plot_training_results()`

**Hiperparámetros importantes:**
- `actor_lr`: 3e-4
- `critic_lr`: 3e-4
- `alpha_lr`: 3e-4 (si auto_tune=True)
- `tau`: 0.005
- `auto_tune`: True (recomendado)
- `target_entropy`: -action_dim (heurística)
- `batch_size`: 256
- `updates_per_step`: 1

**Cuándo usar SAC:**
- **Primera opción** para control continuo
- Robótica real
- Problemas complejos
- Cuando sample efficiency importa
- Cuando exploración es crítica

**Ventajas:**
- Estado del arte en continuous control
- Muy robusto y estable
- Explora mejor que TD3/DDPG
- Auto-tuning reduce hyperparameter search
- Aprende múltiples modos de solución

---

## 📊 Comparación de Algoritmos

### Tabla Comparativa

| Característica | PPO | DDPG | TD3 | SAC |
|----------------|-----|------|-----|-----|
| **Tipo** | On-policy | Off-policy | Off-policy | Off-policy |
| **Política** | Stochastic | Deterministic | Deterministic | Stochastic |
| **Acciones** | Disc/Cont | Continuous | Continuous | Continuous |
| **Sample Efficiency** | Media | Alta | Alta | Alta |
| **Estabilidad** | Alta | Media | Alta | Muy Alta |
| **Simplicidad** | Media | Alta | Media | Media |
| **Exploración** | Entropy bonus | Noise | Noise | Maximum Entropy |
| **Hyperparameters** | Medio | Alto | Medio | Bajo (auto-tune) |
| **Uso en Producción** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Performance Esperada

**Pendulum-v1** (200 episodios):
- PPO: ~-200 reward
- DDPG: ~-200 reward
- TD3: ~-150 reward
- SAC: ~-150 reward (más rápido)

**MountainCarContinuous-v0** (500 episodios):
- PPO: ~90 reward
- DDPG: ~85 reward (inestable)
- TD3: ~90 reward (estable)
- SAC: ~90 reward (más consistente)

---

## 🚀 Guía de Uso

### Quick Start

```python
import gymnasium as gym
from advanced import PPOAgent, TD3Agent, SACAgent

# 1. PPO para CartPole (discreto)
env = gym.make('CartPole-v1')
agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    continuous=False
)
agent.train(env, n_episodes=300)

# 2. TD3 para Pendulum (continuo)
env = gym.make('Pendulum-v1')
agent = TD3Agent(
    state_dim=3,
    action_dim=1,
    max_action=2.0
)
agent.train(env, n_episodes=200)

# 3. SAC para control continuo (SOTA)
env = gym.make('Pendulum-v1')
agent = SACAgent(
    state_dim=3,
    action_dim=1,
    auto_tune=True
)
agent.train(env, n_episodes=150)
```

### Workflow Completo

```python
import gymnasium as gym
from advanced.sac import SACAgent, evaluate_agent, plot_training_results

# 1. Crear ambiente
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 2. Crear agente
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    auto_tune=True,
    hidden_dims=[256, 256]
)

# 3. Entrenar
history = agent.train(
    env=env,
    n_episodes=200,
    warmup_steps=1000,
    print_every=10,
    save_path='sac_pendulum.pth'
)

# 4. Visualizar
plot_training_results(history, 'training.png')

# 5. Evaluar
mean_reward, std_reward = evaluate_agent(agent, env, n_episodes=50)
print(f"Reward: {mean_reward:.2f} ± {std_reward:.2f}")

# 6. Guardar/Cargar
agent.save('final_model.pth')
agent.load('final_model.pth')

env.close()
```

---

## 🎯 Decisión: ¿Qué Algoritmo Usar?

### Diagrama de Decisión

```
¿Acciones continuas o discretas?
├─ Discretas → PPO
└─ Continuas → ¿Necesitas SOTA?
              ├─ Sí → SAC (primera opción)
              └─ No → ¿Necesitas determinismo?
                     ├─ Sí → TD3
                     └─ No → PPO o SAC
```

### Recomendaciones por Caso de Uso

**Robótica Real:**
- Primera opción: **SAC** (robusto, explora bien, stochastic)
- Alternativa: **PPO** (si on-policy es aceptable)

**Juegos:**
- Acciones discretas: **PPO**
- Acciones continuas: **SAC** o **TD3**

**Simulación / Investigación:**
- Baseline: **PPO** (fácil de implementar)
- SOTA: **SAC** (mejores resultados)
- Alternativa: **TD3** (buen balance)

**Prototipado Rápido:**
- **DDPG** (simple, rápido de entrenar)
- **PPO** (si necesitas estabilidad)

**Control de Procesos Industriales:**
- **TD3** o **SAC** (continuos, robustos)

---

## 💡 Tips de Entrenamiento

### 1. Hyperparameters Generales

**Learning Rates:**
- Actor: 3e-4 (PPO, TD3, SAC)
- Critic: 3e-4 o 1e-3 (TD3, SAC)
- Empieza con estos, ajusta si inestable

**Network Size:**
- Default: [256, 256] funciona bien
- Problemas simples: [64, 64]
- Problemas complejos: [400, 300] o [512, 512]

**Batch Size:**
- PPO: 64 (on-policy, menos datos)
- DDPG/TD3/SAC: 256 (off-policy, más estable)

**Buffer Size:**
- Problemas simples: 100K
- Problemas complejos: 1M

### 2. Debugging

**Si entrenamiento no converge:**
1. Verifica que ambiente funcione correctamente
2. Reduce learning rate (x0.3)
3. Aumenta warmup steps
4. Revisa que recompensas estén normalizadas

**Si es inestable:**
1. Aumenta batch size
2. Reduce learning rate
3. Activa gradient clipping (ya incluido)
4. Para PPO: reduce epsilon_clip

**Si explora poco:**
1. PPO: aumenta entropy_coef
2. SAC: verifica que auto_tune=True
3. DDPG/TD3: aumenta exploration_noise

### 3. Mejores Prácticas

✅ **Hacer:**
- Usa warmup period para llenar buffer
- Normaliza observaciones si tienen escalas muy diferentes
- Normaliza rewards si son muy variables
- Guarda checkpoints regularmente
- Evalúa sin ruido/deterministic
- Usa tensorboard para monitorear métricas

❌ **Evitar:**
- Cambiar muchos hyperparameters a la vez
- Entrenar sin warmup (off-policy)
- Ignorar warnings de NaN/Inf
- Usar batch size muy pequeño (off-policy)
- Updates demasiado frecuentes (PPO)

---

## 📚 Referencias

### Papers Originales

1. **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
   Schulman et al., 2017

2. **DDPG**: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
   Lillicrap et al., 2016

3. **TD3**: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
   Fujimoto et al., 2018

4. **SAC**: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
   Haarnoja et al., 2019

### Recursos Adicionales

- **Spinning Up in Deep RL** (OpenAI): Excelente tutorial
- **Stable-Baselines3**: Implementaciones de referencia
- **CleanRL**: Implementaciones simples y limpias
- **RLlib** (Ray): Para entrenamiento distribuido

---

## 🧪 Testing

Para verificar que las implementaciones funcionan:

```bash
# Test PPO
python 03_deep_rl/advanced/ppo.py

# Test DDPG
python 03_deep_rl/advanced/ddpg.py

# Test TD3
python 03_deep_rl/advanced/td3.py

# Test SAC
python 03_deep_rl/advanced/sac.py
```

Cada script incluye ejemplos completos en su función `main()`.

---

## 📝 Notas Finales

Estos algoritmos son **production-ready** y siguen las mejores prácticas:

✅ Type hints completos
✅ Docstrings en español
✅ Gradient clipping
✅ Soft/hard updates
✅ Save/load functionality
✅ Plotting utilities
✅ Evaluation functions
✅ Comprehensive examples

**Total:** ~3700 líneas de código de calidad profesional.

---

**Autor:** MARK-126
**Versión:** 1.0.0
**Última actualización:** 2025
