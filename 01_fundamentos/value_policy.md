# Value Functions y Políticas

## Introducción

Las **value functions** (funciones de valor) son el corazón del reinforcement learning. Nos dicen qué tan bueno es estar en un estado o tomar una acción. Las **políticas** nos dicen cómo actuar. Entender la relación entre ambas es crucial para dominar RL.

## Funciones de Valor

### State-Value Function: V^π(s)

**Definición**: El valor esperado del retorno empezando en estado s y siguiendo política π.

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... | S_t = s]
```

**Interpretación**: "¿Qué tan bueno es estar en este estado si sigo esta política?"

### Action-Value Function: Q^π(s,a)

**Definición**: El valor esperado del retorno empezando en estado s, tomando acción a, y luego siguiendo política π.

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E_π[R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... | S_t = s, A_t = a]
```

**Interpretación**: "¿Qué tan bueno es tomar esta acción en este estado y luego seguir la política?"

### Advantage Function: A^π(s,a)

**Definición**: Cuánto mejor es tomar acción a comparado con el promedio en estado s.

```
A^π(s,a) = Q^π(s,a) - V^π(s)
```

**Interpretación**:
- A^π(s,a) > 0: La acción a es mejor que el promedio
- A^π(s,a) < 0: La acción a es peor que el promedio
- A^π(s,a) = 0: La acción a es promedio

**Uso**: Métodos Actor-Critic modernos (A3C, PPO)

## Políticas

### Política Determinista

Mapea cada estado a una acción específica:

```
π: S → A
π(s) = a
```

**Ejemplo**:
```python
policy = {
    'state1': 'up',
    'state2': 'right',
    'state3': 'down'
}
```

### Política Estocástica

Mapea estados a distribuciones de probabilidad sobre acciones:

```
π: S × A → [0,1]
π(a|s) = P(A_t = a | S_t = s)
```

**Ejemplo**:
```python
policy = {
    'state1': {'up': 0.7, 'down': 0.3},
    'state2': {'left': 0.5, 'right': 0.5},
    'state3': {'up': 0.1, 'down': 0.9}
}
```

### ¿Por qué Políticas Estocásticas?

1. **Exploración**: Permiten explorar naturalmente
2. **Optimalidad en POMDPs**: A veces son necesarias para optimalidad
3. **Gradiente de política**: Permiten calcular gradientes
4. **Juegos**: Impredecibilidad puede ser ventajosa (rock-paper-scissors)

## Relaciones entre V, Q y π

### 1. De Q a V

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)
```

Para política determinista π(s) = a*:
```
V^π(s) = Q^π(s, a*)
```

### 2. De V a Q

```
Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')]
```

### 3. Política Greedy desde V

```
π_greedy(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
```

**Problema**: Requiere conocer modelo (P y R)

### 4. Política Greedy desde Q

```
π_greedy(s) = argmax_a Q(s,a)
```

**Ventaja**: No requiere modelo, solo valores Q

## Evaluación de Política (Policy Evaluation)

**Objetivo**: Calcular V^π para una política π dada

### Método Iterativo

Inicializar V(s) arbitrariamente para todo s

Repetir hasta convergencia:
```
V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V_k(s')]
```

### Ejemplo en Código

```python
def policy_evaluation(policy, env, gamma=0.9, theta=1e-6):
    """
    Evalúa una política dada
    
    Args:
        policy: Política a evaluar π(a|s)
        env: Ambiente MDP
        gamma: Factor de descuento
        theta: Umbral de convergencia
    
    Returns:
        V: Value function V^π
    """
    V = {s: 0 for s in env.states}
    
    while True:
        delta = 0
        
        for s in env.states:
            v = V[s]
            
            # Aplicar ecuación de Bellman
            new_v = 0
            for a in env.actions:
                for s_next in env.states:
                    prob_transition = env.P(s_next, s, a)
                    reward = env.R(s, a, s_next)
                    new_v += policy[s][a] * prob_transition * (reward + gamma * V[s_next])
            
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V
```

## Mejora de Política (Policy Improvement)

**Objetivo**: Mejorar política π para obtener π' ≥ π

### Teorema de Mejora de Política

Si para todo s:
```
Q^π(s, π'(s)) ≥ V^π(s)
```

Entonces:
```
V^{π'}(s) ≥ V^π(s) para todo s
```

### Política Greedy

La política greedy respecto a V^π siempre es una mejora:

```
π'(s) = argmax_a Q^π(s,a)
      = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')]
```

### Ejemplo en Código

```python
def policy_improvement(V, env, gamma=0.9):
    """
    Mejora la política usando greedy sobre V
    
    Args:
        V: Value function actual
        env: Ambiente MDP
        gamma: Factor de descuento
    
    Returns:
        policy: Nueva política mejorada
        policy_stable: Si la política no cambió
    """
    policy = {s: None for s in env.states}
    policy_stable = True
    
    for s in env.states:
        old_action = policy[s]
        
        # Encontrar mejor acción
        action_values = {}
        for a in env.actions:
            value = 0
            for s_next in env.states:
                prob = env.P(s_next, s, a)
                reward = env.R(s, a, s_next)
                value += prob * (reward + gamma * V[s_next])
            action_values[a] = value
        
        # Greedy
        policy[s] = max(action_values, key=action_values.get)
        
        if old_action != policy[s]:
            policy_stable = False
    
    return policy, policy_stable
```

## Funciones de Valor Óptimas

### V*(s): Optimal State-Value Function

```
V*(s) = max_π V^π(s)
      = max_a Q*(s,a)
```

**Interpretación**: El mejor valor que podemos obtener desde estado s.

### Q*(s,a): Optimal Action-Value Function

```
Q*(s,a) = max_π Q^π(s,a)
```

**Interpretación**: El mejor valor que podemos obtener tomando acción a en estado s y actuando óptimamente después.

### Propiedades

1. **Unicidad**: V* y Q* son únicos
2. **Consistencia**: Satisfacen ecuaciones de Bellman óptimas
3. **Política óptima**: π*(s) = argmax_a Q*(s,a)

## Política Óptima

Una política π* es óptima si:
```
V^{π*}(s) = V*(s) para todo s
```

### Propiedades de π*

1. **Existe**: Siempre existe al menos una política óptima
2. **Múltiples**: Puede haber múltiples políticas óptimas
3. **Determinista**: Siempre existe una π* determinista (para MDPs)
4. **Parcial ordenamiento**: Políticas se pueden ordenar por dominancia

### Obtención de π* desde Q*

```
π*(s) = argmax_a Q*(s,a)
```

**Ventaja clave**: No necesitamos modelo del ambiente

## Exploración vs Explotación

### Explotación
Seleccionar la acción que creemos es mejor según nuestro conocimiento actual:
```
a = argmax_a Q(s,a)
```

### Exploración
Probar acciones que pueden no ser óptimas para obtener más información:
```
a = acción_aleatoria()
```

### Estrategias de Exploración

#### 1. ε-greedy

```python
def epsilon_greedy(Q, s, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(actions)  # Exploración
    else:
        return argmax(Q[s])  # Explotación
```

Parámetros típicos:
- Inicio: ε = 1.0 (100% exploración)
- Decaimiento: ε = ε * 0.995 cada episodio
- Mínimo: ε_min = 0.01

#### 2. Softmax / Boltzmann

```python
def softmax_policy(Q, s, tau=1.0):
    """
    tau: temperatura
      - tau → 0: política greedy
      - tau → ∞: política uniforme
    """
    probs = np.exp(Q[s] / tau) / np.sum(np.exp(Q[s] / tau))
    return np.random.choice(actions, p=probs)
```

#### 3. Upper Confidence Bound (UCB)

```python
def ucb_action(Q, N, s, c=2):
    """
    Q: valores estimados
    N: número de veces que cada acción ha sido seleccionada
    c: parámetro de exploración
    """
    ucb_values = Q[s] + c * np.sqrt(np.log(sum(N[s])) / N[s])
    return argmax(ucb_values)
```

## On-Policy vs Off-Policy

### On-Policy
Aprende sobre la política que está ejecutando.

**Ejemplo**: SARSA
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```
donde a' es elegida por la política actual

**Ventajas**: Más estable, mejor convergencia
**Desventajas**: Menos eficiente en datos

### Off-Policy
Aprende sobre una política diferente a la que ejecuta.

**Ejemplo**: Q-Learning
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```
Aprende sobre política greedy mientras ejecuta ε-greedy

**Ventajas**: Más eficiente en datos, puede aprender de datos antiguos
**Desventajas**: Menos estable, requiere importance sampling en casos generales

## Visualización de Value Functions

### GridWorld 4x4

```
V*(s) para cada celda:

 0.0  -1.0  -2.0  -3.0
-1.0  -2.0  -3.0  -2.0
-2.0  -3.0  -2.0  -1.0
-3.0  -2.0  -1.0   0.0

Política óptima π*:
→    →     →     ↓
↓    -     -     ↓
↓    -     -     ↓
→    →     →     GOAL
```

## Implementación Completa: Policy Iteration

```python
def policy_iteration(env, gamma=0.9, theta=1e-6):
    """
    Policy Iteration: Alterna entre evaluación y mejora
    
    Returns:
        policy: Política óptima
        V: Value function óptima
    """
    # Inicializar política arbitraria
    policy = {s: random.choice(env.actions) for s in env.states}
    
    while True:
        # 1. Policy Evaluation
        V = policy_evaluation(policy, env, gamma, theta)
        
        # 2. Policy Improvement
        policy, policy_stable = policy_improvement(V, env, gamma)
        
        # Si la política no cambió, es óptima
        if policy_stable:
            break
    
    return policy, V
```

## Ejercicios

### Ejercicio 1
Para un GridWorld 2x2 con objetivo en (1,1), calcula V^π manualmente para una política que siempre va hacia derecha o abajo.

### Ejercicio 2
Implementa ε-greedy con decaimiento exponencial. ¿Cómo afecta la tasa de decaimiento al aprendizaje?

### Ejercicio 3
Demuestra que si Q^π(s, π'(s)) ≥ V^π(s) para todo s, entonces π' ≥ π.

## Conceptos Clave

1. **V y Q**: Cuantifican qué tan buenos son estados y acciones
2. **Políticas**: Determinan cómo actuar
3. **Greedy**: Política que maximiza valor
4. **Exploración**: Necesaria para descubrir valor verdadero
5. **On/Off-Policy**: Diferencia entre qué aprendemos y qué hacemos

## Próximos Pasos

Con este conocimiento, estás listo para:
1. Estudiar [Programación Dinámica](../02_algoritmos_clasicos/dynamic_programming/) - Métodos exactos
2. Aprender [Monte Carlo Methods](../02_algoritmos_clasicos/monte_carlo/) - Métodos de muestreo
3. Explorar [TD Learning](../02_algoritmos_clasicos/temporal_difference/) - Bootstrapping

## Referencias

- Sutton & Barto, Chapters 3-4: MDPs and Dynamic Programming
- Silver, Lecture 2: Markov Decision Processes
- Bertsekas: Dynamic Programming and Optimal Control
