# Conceptos de Optimización para Deep Reinforcement Learning

## 🎯 Objetivo

En Deep RL, necesitamos **optimizar** (encontrar mejores) políticas, value functions, y redes neuronales. Esta guía cubre los algoritmos de optimización más importantes.

---

## 1. El Problema de Optimización

### 1.1 Definición

**Objetivo**: Encontrar parámetros θ que minimicen (o maximicen) una función objetivo J(θ).

**Minimización** (ej: loss de red neuronal):
```
θ* = argmin_θ L(θ)
```

**Maximización** (ej: retorno esperado en RL):
```
θ* = argmax_θ J(θ)
```

**Truco**: Maximizar J(θ) = Minimizar -J(θ)

### 1.2 Ejemplos en RL

**Supervised Learning** (para comparación):
```
Minimizar: L(θ) = (1/N) Σᵢ (f_θ(xᵢ) - yᵢ)²
```

**Policy Gradient**:
```
Maximizar: J(θ) = E_π[Σ γᵗrₜ]
```

**DQN**:
```
Minimizar: L(θ) = E[(r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]
```

---

## 2. Gradient Descent (Descenso por Gradiente)

### 2.1 Idea Básica

Si quieres **minimizar** L(θ), muévete en dirección opuesta al gradiente:

```
θ_{t+1} = θ_t - α·∇_θ L(θ_t)
```

Donde:
- **α** (alpha): learning rate (paso)
- **∇_θ L**: gradiente (dirección de mayor crecimiento)

**Analogía**: Bajar una montaña en la niebla siguiendo la pendiente más inclinada.

### 2.2 Tipos de Gradient Descent

#### Batch Gradient Descent

Usa **todos** los datos para calcular gradiente:
```
θ = θ - α·(1/N)Σᵢ₌₁ᴺ ∇_θ L(θ; xᵢ, yᵢ)
```

**Ventajas**: Gradiente exacto, convergencia suave
**Desventajas**: Lento para datasets grandes

#### Stochastic Gradient Descent (SGD)

Usa **un solo** ejemplo cada vez:
```
θ = θ - α·∇_θ L(θ; xᵢ, yᵢ)
```

**Ventajas**: Rápido, puede escapar mínimos locales
**Desventajas**: Ruidoso, inestable

#### Mini-Batch Gradient Descent

Usa **mini-batch** de B ejemplos:
```
θ = θ - α·(1/B)Σᵢ₌₁ᴮ ∇_θ L(θ; xᵢ, yᵢ)
```

**Ventajas**: Balance entre velocidad y estabilidad
**Uso típico**: B = 32, 64, 128, 256

**Código**:
```python
import numpy as np

# Datos
X = np.random.randn(1000, 10)  # 1000 ejemplos, 10 features
y = np.random.randn(1000, 1)

# Parámetros
theta = np.random.randn(10, 1)
alpha = 0.01
batch_size = 32

# Mini-batch SGD
for epoch in range(100):
    # Shuffle datos
    indices = np.random.permutation(len(X))

    for i in range(0, len(X), batch_size):
        # Mini-batch
        batch_indices = indices[i:i+batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        # Forward
        y_pred = X_batch @ theta
        error = y_pred - y_batch

        # Gradiente
        gradient = (2 / batch_size) * X_batch.T @ error

        # Update
        theta -= alpha * gradient
```

### 2.3 Learning Rate (α)

El learning rate controla el tamaño del paso.

**Demasiado grande**:
```
θ → overshooting → divergencia
```

**Demasiado pequeño**:
```
θ → convergencia muy lenta
```

**Valores típicos**: 0.001, 0.0001, 0.01

**Visualización**:
```python
import matplotlib.pyplot as plt

def f(x):
    return x**2

def df(x):
    return 2*x

x = 5.0

# Diferentes learning rates
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
learning_rates = [0.1, 0.5, 1.1]

for idx, alpha in enumerate(learning_rates):
    x_history = [x]
    x_temp = x

    for _ in range(20):
        x_temp = x_temp - alpha * df(x_temp)
        x_history.append(x_temp)

    axes[idx].plot(x_history, [f(xi) for xi in x_history], 'o-')
    axes[idx].set_title(f'α = {alpha}')
    axes[idx].set_xlabel('Iteración')
    axes[idx].set_ylabel('f(x)')
    axes[idx].grid()

plt.tight_layout()
plt.show()
```

---

## 3. Momentum (Momento)

### 3.1 Problema de SGD Vanilla

SGD puro puede:
- Oscilar en valles estrechos
- Atascarse en plateaus
- Ser muy ruidoso

### 3.2 SGD con Momentum

Agrega "inercia" acumulando gradientes pasados:

```
v_t = β·v_{t-1} + ∇_θ L(θ_t)
θ_{t+1} = θ_t - α·v_t
```

Donde:
- **v**: velocity (velocidad acumulada)
- **β** (beta): coeficiente de momento (típicamente 0.9)

**Analogía**: Una bola rodando cuesta abajo acumula velocidad.

**Ventajas**:
- Acelera en direcciones consistentes
- Reduce oscilaciones
- Ayuda a escapar mínimos locales

**Código**:
```python
# SGD con Momentum
v = np.zeros_like(theta)
beta = 0.9
alpha = 0.01

for epoch in range(100):
    gradient = compute_gradient(theta)

    # Actualizar velocidad
    v = beta * v + gradient

    # Actualizar parámetros
    theta -= alpha * v
```

---

## 4. Adam (Adaptive Moment Estimation)

### 4.1 Motivación

Adam combina:
- **Momentum** (first moment)
- **RMSprop** (second moment - adapta learning rate por parámetro)

**Es el optimizador más popular en Deep RL.**

### 4.2 Algoritmo

```
m_t = β₁·m_{t-1} + (1-β₁)·∇_θ L          # First moment (momentum)
v_t = β₂·v_{t-1} + (1-β₂)·(∇_θ L)²       # Second moment (varianza)

m̂_t = m_t / (1-β₁ᵗ)                      # Bias correction
v̂_t = v_t / (1-β₂ᵗ)

θ_{t+1} = θ_t - α·m̂_t / (√v̂_t + ε)
```

**Hyperparámetros por defecto**:
- α = 0.001
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8

**Ventajas**:
- Adapta learning rate por parámetro
- Funciona bien out-of-the-box
- Converge rápido

**Código**:
```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step

    def update(self, theta, gradient):
        # Inicializar momentos
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)

        self.t += 1

        # Actualizar momentos
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Actualizar parámetros
        theta -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return theta

# Uso
optimizer = Adam(learning_rate=0.001)

for epoch in range(100):
    gradient = compute_gradient(theta)
    theta = optimizer.update(theta, gradient)
```

### 4.3 Comparación Visual

```python
# Comparar SGD vs Momentum vs Adam
def rosenbrock(x, y):
    """Función de prueba difícil de optimizar"""
    return (1 - x)**2 + 100 * (y - x**2)**2

# Implementar y graficar trayectorias...
```

---

## 5. Learning Rate Schedules

### 5.1 ¿Por Qué Decaer Learning Rate?

- **Inicio**: α grande para explorar
- **Final**: α pequeño para convergencia fina

### 5.2 Tipos de Schedules

#### Step Decay

```
α_t = α_0 · γ^(t // k)
```

Reduce α por factor γ cada k epochs.

**Ejemplo**:
```python
def step_decay(epoch, alpha_0=0.1, drop=0.5, epochs_drop=100):
    return alpha_0 * (drop ** (epoch // epochs_drop))

# Epoch 0-99:    α = 0.1
# Epoch 100-199: α = 0.05
# Epoch 200-299: α = 0.025
```

#### Exponential Decay

```
α_t = α_0 · e^(-λt)
```

**Código**:
```python
def exponential_decay(epoch, alpha_0=0.1, decay_rate=0.01):
    return alpha_0 * np.exp(-decay_rate * epoch)
```

#### Polynomial Decay

```
α_t = α_0 · (1 - t/T)^p
```

**Código**:
```python
def polynomial_decay(epoch, max_epochs, alpha_0=0.1, power=1.0):
    return alpha_0 * (1 - epoch / max_epochs) ** power
```

#### Cosine Annealing

```
α_t = α_min + (α_max - α_min) · (1 + cos(πt/T)) / 2
```

**Código**:
```python
def cosine_annealing(epoch, max_epochs, alpha_max=0.1, alpha_min=0.0):
    return alpha_min + (alpha_max - alpha_min) * \
           (1 + np.cos(np.pi * epoch / max_epochs)) / 2
```

**Visualizar**:
```python
epochs = np.arange(0, 1000)

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.plot(epochs, [step_decay(e) for e in epochs])
plt.title('Step Decay')

plt.subplot(1, 4, 2)
plt.plot(epochs, [exponential_decay(e) for e in epochs])
plt.title('Exponential Decay')

plt.subplot(1, 4, 3)
plt.plot(epochs, [polynomial_decay(e, 1000) for e in epochs])
plt.title('Polynomial Decay')

plt.subplot(1, 4, 4)
plt.plot(epochs, [cosine_annealing(e, 1000) for e in epochs])
plt.title('Cosine Annealing')

plt.tight_layout()
plt.show()
```

---

## 6. Gradient Clipping

### 6.1 Problema: Exploding Gradients

En redes profundas, gradientes pueden **explotar** (volverse muy grandes).

### 6.2 Solución: Clipear Gradientes

Limita la magnitud del gradiente:

**Clip by value**:
```python
gradient = np.clip(gradient, -clip_value, clip_value)
```

**Clip by norm** (más común):
```python
def clip_gradient_norm(gradient, max_norm=1.0):
    """Clipea gradiente si su norma excede max_norm"""
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        gradient = gradient * (max_norm / norm)
    return gradient

# Uso
gradient = compute_gradient(theta)
gradient = clip_gradient_norm(gradient, max_norm=1.0)
theta -= alpha * gradient
```

**En PyTorch**:
```python
import torch

# Calcular gradientes
loss.backward()

# Clipear antes de update
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Actualizar pesos
optimizer.step()
```

---

## 7. Optimización en Deep RL

### 7.1 Policy Gradient

**Objetivo**: Maximizar J(θ) = E_π[Σ γᵗrₜ]

**Update** (gradient ascent):
```
θ = θ + α·∇_θ J(θ)
```

**Policy Gradient Theorem**:
```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s)·Q^π(s,a)]
```

**Código**:
```python
# REINFORCE algorithm
def train_policy_gradient(env, policy, optimizer, num_episodes=1000):
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []

        # Generar episodio
        state = env.reset()
        done = False

        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Calcular retornos
        returns = compute_returns(rewards, gamma=0.99)

        # Policy gradient update
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            G = returns[t]

            # Log probability
            log_prob = policy.log_prob(state, action)

            # Loss (negative para maximizar)
            loss = -log_prob * G

            # Backprop
            loss.backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()
```

### 7.2 Q-Learning con Redes Neuronales (DQN)

**Objetivo**: Minimizar L(θ) = E[(target - Q(s,a;θ))²]

**Código**:
```python
# DQN training loop
def train_dqn(env, q_network, target_network, optimizer,
              replay_buffer, batch_size=32):

    # Sample batch
    states, actions, rewards, next_states, dones = \
        replay_buffer.sample(batch_size)

    # Current Q-values
    q_values = q_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1))

    # Target Q-values (con target network)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

    # Loss
    loss = F.mse_loss(q_values.squeeze(), targets)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    optimizer.step()
```

### 7.3 Actor-Critic

**Dos redes**:
- **Actor** π_θ: genera acciones
- **Critic** V_φ: evalúa estados

**Updates**:
```
Actor:  θ = θ + α_actor·∇_θ log π_θ(a|s)·A(s,a)
Critic: φ = φ - α_critic·∇_φ(V_φ(s) - target)²
```

Donde A(s,a) = Q(s,a) - V(s) es la advantage function.

---

## 8. Hiperparámetros: Cómo Elegir

### 8.1 Learning Rate (α)

**Heurística**:
- **Policy Gradient**: 1e-4 a 1e-3
- **DQN**: 1e-4 a 1e-3
- **Actor-Critic**: 1e-4 (actor), 1e-3 (critic)

**Grid search**:
```python
alphas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

for alpha in alphas:
    rewards = train(alpha=alpha)
    print(f"α={alpha}: Mean reward = {np.mean(rewards)}")
```

### 8.2 Batch Size

**Valores típicos**: 32, 64, 128, 256

**Trade-off**:
- Pequeño: Más ruidoso, más actualizaciones
- Grande: Más estable, menos actualizaciones

### 8.3 Optimizador

**Recomendaciones**:
- **Empezar**: Adam con defaults
- **Fine-tuning**: SGD con momentum
- **Rápido**: RMSprop

### 8.4 Gamma (descuento)

**Valores típicos**: 0.95, 0.99, 0.999

**Heurística**:
- Episodios cortos: γ = 0.95
- Episodios largos: γ = 0.99 o 0.999

---

## 9. Trucos Prácticos

### 9.1 Normalización de Inputs

```python
# Normalizar estados
state_mean = np.mean(states, axis=0)
state_std = np.std(states, axis=0)

state_normalized = (state - state_mean) / (state_std + 1e-8)
```

### 9.2 Reward Scaling

```python
# Escalar recompensas a rango [-1, 1] o [0, 1]
reward_normalized = reward / reward_max

# O estandarizar
reward_normalized = (reward - reward_mean) / reward_std
```

### 9.3 Advantage Normalization

```python
# Normalizar advantages para estabilidad
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 9.4 Target Network Updates

```python
# Actualización soft (polyak averaging)
tau = 0.005

for target_param, param in zip(target_net.parameters(), q_net.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### 9.5 Entropy Bonus (Policy Gradient)

```python
# Agregar entropía para fomentar exploración
entropy = -torch.sum(probs * torch.log(probs + 1e-8))
loss = policy_loss - entropy_coefficient * entropy
```

---

## 10. Debugging Optimization

### 10.1 Métricas a Monitorear

```python
# Durante entrenamiento, graficar:
- Learning rate
- Loss value
- Gradient norm
- Weight norm
- Reward por episodio
```

### 10.2 Verificar Gradientes

```python
# Verificar que gradientes no sean NaN o Inf
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm = {grad_norm:.4f}")

        if np.isnan(grad_norm) or np.isinf(grad_norm):
            print(f"WARNING: {name} has NaN or Inf gradients!")
```

### 10.3 Loss Landscape Visualization

```python
# Graficar loss vs epochs
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid()
plt.show()
```

---

## 11. Ejercicios Prácticos

### Ejercicio 1: Implementar Momentum

```python
def sgd_momentum(theta, gradients, velocity, alpha=0.01, beta=0.9):
    """
    Implementa SGD con momentum

    Args:
        theta: Parámetros actuales
        gradients: Lista de gradientes por mini-batch
        velocity: Velocidad actual
        alpha: Learning rate
        beta: Momentum coefficient

    Returns:
        theta actualizado, velocity actualizado
    """
    # TU CÓDIGO AQUÍ
    pass
```

<details>
<summary>Ver solución</summary>

```python
def sgd_momentum(theta, gradients, velocity, alpha=0.01, beta=0.9):
    for gradient in gradients:
        velocity = beta * velocity + gradient
        theta -= alpha * velocity

    return theta, velocity
```
</details>

### Ejercicio 2: Comparar Optimizadores

Implementa experimento que compare SGD, Momentum y Adam en una función simple.

---

## 12. Cheat Sheet

| Optimizador | Update Rule | Pros | Contras | Uso en RL |
|-------------|-------------|------|---------|-----------|
| **SGD** | θ -= α·∇L | Simple | Lento, ruidoso | Raro |
| **SGD + Momentum** | v=βv+∇L; θ-=αv | Más rápido | Requiere tuning | Ocasional |
| **Adam** | Adaptivo | Funciona bien out-of-box | Puede overshoot | **MÁS COMÚN** |
| **RMSprop** | Adaptivo | Bueno para RNNs | Menos usado | Ocasional |

---

## 13. Recursos Adicionales

### Papers
- Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization"
- Sutskever et al. (2013). "On the importance of initialization and momentum"

### Tutoriales
- [CS231n - Optimization](http://cs231n.github.io/neural-networks-3/)
- [Distill.pub - Why Momentum Works](https://distill.pub/2017/momentum/)

### Herramientas
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [TensorFlow Optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)

---

## 14. Autoevaluación

¿Puedes explicar...?

- [ ] Diferencia entre SGD, mini-batch SGD, y batch GD
- [ ] Cómo funciona momentum
- [ ] Por qué Adam es popular
- [ ] Cuándo usar gradient clipping
- [ ] Cómo elegir learning rate

Si respondiste todo, ¡estás listo para Deep RL! 🎉

---

## Próximos Pasos

1. **[README de Prerequisites](00_README.md)** - Resumen de toda la preparación
2. **[Fundamentos de RL](../01_fundamentos/introduccion.md)** - ¡Empezar con RL!

¡Felicitaciones por completar todos los prerequisitos! 🚀
