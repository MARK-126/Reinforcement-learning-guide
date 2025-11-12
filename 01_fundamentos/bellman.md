# Ecuaciones de Bellman

## Introducción

Las **Ecuaciones de Bellman** son fundamentales en reinforcement learning. Proporcionan relaciones recursivas que expresan la consistencia entre el valor de un estado (o par estado-acción) y los valores de sus sucesores.

Estas ecuaciones son la base de muchos algoritmos de RL y nos permiten calcular funciones de valor de manera eficiente.

## Ecuación de Bellman para V^π

### Forma Intuitiva

El valor de un estado es igual a:
```
Recompensa inmediata + Valor descontado del siguiente estado
```

### Forma Matemática

```
V^π(s) = E_π[R_{t+1} + γ V^π(S_{t+1}) | S_t = s]
```

Expandiendo la esperanza:

```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')]
```

### Explicación Componente por Componente

1. **π(a|s)**: Probabilidad de tomar acción a en estado s
2. **P(s'|s,a)**: Probabilidad de transición a s' dado s y a
3. **R(s,a,s')**: Recompensa inmediata
4. **γ V^π(s')**: Valor futuro descontado

## Ecuación de Bellman para Q^π

### Forma Matemática

```
Q^π(s,a) = E_π[R_{t+1} + γ Q^π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
```

Expandiendo:

```
Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ Σ_{a'} π(a'|s') Q^π(s',a')]
```

## Relación entre V^π y Q^π

### De Q a V

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)
```

Si la política es determinista: π(s) = a
```
V^π(s) = Q^π(s, π(s))
```

### De V a Q

```
Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')]
```

## Ecuación de Bellman Óptima

### Para V* (Optimal State-Value)

```
V*(s) = max_a Q*(s,a)
      = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V*(s')]
```

**Interpretación**: El valor óptimo de un estado es el mejor valor acción-estado que podemos obtener desde ese estado.

### Para Q* (Optimal Action-Value)

```
Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ max_{a'} Q*(s',a')]
```

**Interpretación**: El valor óptimo de un par estado-acción es la recompensa esperada más el valor óptimo del mejor siguiente par estado-acción.

### Política Óptima desde V*

```
π*(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V*(s')]
```

### Política Óptima desde Q*

```
π*(s) = argmax_a Q*(s,a)
```

**Ventaja de Q***: No necesitamos conocer el modelo (P y R) para actuar óptimamente.

## Ejemplo Numérico: GridWorld

Consideremos un GridWorld 3x3 simple:

```
[S] [ ] [G]
[ ] [X] [ ]
[ ] [ ] [ ]

S = Start (estado inicial)
G = Goal (recompensa +1)
X = Trap (recompensa -1)
```

### Parámetros
- γ = 0.9
- Acciones: {↑, ↓, ←, →}
- Política: equiprobable (25% cada acción)
- Recompensa: +1 en G, -1 en X, -0.01 en otros

### Cálculo de V^π para estado inicial S

Supongamos que desde S:
- ↑ → muro (queda en S)
- ↓ → estado (1,0)
- ← → muro (queda en S)
- → → estado (0,1)

```
V^π(S) = 0.25 * [-0.01 + 0.9*V^π(S)]      # ↑
       + 0.25 * [-0.01 + 0.9*V^π(1,0)]    # ↓
       + 0.25 * [-0.01 + 0.9*V^π(S)]      # ←
       + 0.25 * [-0.01 + 0.9*V^π(0,1)]    # →
```

Este sistema de ecuaciones se resuelve iterativamente.

## Backup Diagrams

Los **diagramas de backup** visualizan las ecuaciones de Bellman:

### Backup para V^π

```
    s
   /|\\ 
  a₁a₂a₃  (acciones según π)
  / | \\
 s₁ s₂ s₃  (estados siguientes)
```

### Backup para Q^π

```
  (s,a)
   /|\\
  s₁s₂s₃  (estados siguientes)
  /|\ 
 a₁a₂a₃  (acciones siguientes según π)
```

### Backup para V*

```
    s
   MAX
   /|\\
  a₁a₂a₃  (todas las acciones)
  /|\\
 ...
```

## Propiedades Importantes

### 1. Punto Fijo
Las ecuaciones de Bellman tienen un **punto fijo único** para cualquier política π:
```
V^π = T^π V^π
```
Donde T^π es el operador de Bellman.

### 2. Contracción
El operador de Bellman óptimo T* es una **contracción** en la norma max:
```
||T* V - T* U||_∞ ≤ γ ||V - U||_∞
```

Esto garantiza convergencia a V*.

### 3. Optimalidad
Si V satisface la ecuación de Bellman óptima, entonces V = V* y la política greedy respecto a V es óptima.

## Uso en Algoritmos

Las ecuaciones de Bellman son la base de muchos algoritmos:

### Dynamic Programming
- **Policy Evaluation**: Resuelve V^π iterativamente
- **Policy Iteration**: Usa ecuación de Bellman para π
- **Value Iteration**: Usa ecuación de Bellman óptima

### Temporal Difference Learning
- **TD(0)**: Aproxima expectativa en ecuación de Bellman
- **Q-Learning**: Aproxima ecuación de Bellman óptima para Q
- **SARSA**: Aproxima ecuación de Bellman para Q^π

### Deep RL
- **DQN**: Minimiza error entre Q y objetivo de Bellman
- **Actor-Critic**: Usa ecuación de Bellman para crítico

## Bellman Expectation vs Bellman Optimality

| Aspecto | Expectation | Optimality |
|---------|-------------|------------|
| Para | Política específica π | Política óptima π* |
| Operador | E_π | max_a |
| Resultado | V^π o Q^π | V* o Q* |
| Unicidad | Único para cada π | Único globalmente |
| Uso | Evaluación de política | Encontrar política óptima |

## Forma Matricial (Espacios Finitos)

Para MDPs con estados finitos, podemos escribir en forma matricial:

```
V^π = R^π + γ P^π V^π
```

Donde:
- V^π: Vector |S| × 1
- R^π: Vector de recompensas esperadas
- P^π: Matriz de transición |S| × |S|

Solución directa:
```
V^π = (I - γ P^π)^{-1} R^π
```

## Ejemplo en Código

```python
import numpy as np

def bellman_update_v(states, V, policy, transition_prob, reward, gamma):
    """
    Actualización de Bellman para V^π
    
    Args:
        states: Lista de estados
        V: Valores actuales (dict)
        policy: Política π(a|s) (dict)
        transition_prob: P(s'|s,a) (dict)
        reward: R(s,a,s') (dict)
        gamma: Factor de descuento
    
    Returns:
        Nuevos valores V
    """
    V_new = {}
    
    for s in states:
        v = 0
        # Suma sobre acciones
        for a, prob_a in policy[s].items():
            # Suma sobre estados siguientes
            for s_next, prob_s in transition_prob[s][a].items():
                r = reward[s][a][s_next]
                v += prob_a * prob_s * (r + gamma * V[s_next])
        V_new[s] = v
    
    return V_new

def bellman_optimality_update(states, V, actions, transition_prob, reward, gamma):
    """
    Actualización de Bellman óptima para V*
    """
    V_new = {}
    
    for s in states:
        # max sobre acciones
        max_value = float('-inf')
        
        for a in actions:
            value = 0
            # Suma sobre estados siguientes
            for s_next, prob_s in transition_prob[s][a].items():
                r = reward[s][a][s_next]
                value += prob_s * (r + gamma * V[s_next])
            
            max_value = max(max_value, value)
        
        V_new[s] = max_value
    
    return V_new
```

## Ejercicios

### Ejercicio 1: Derivación
Deriva la ecuación de Bellman para Q^π empezando desde su definición.

### Ejercicio 2: GridWorld
Para el GridWorld de 2x2 con:
- Estados: {s₁, s₂, s₃, s₄}
- s₄ es terminal con recompensa +1
- Todas las demás transiciones dan -0.1
- γ = 0.9

Calcula V^π para una política uniforme después de una iteración empezando con V(s) = 0 para todo s.

### Ejercicio 3: Implementación
Implementa las funciones de backup de Bellman para un MDP simple de tu elección.

## Conceptos Clave para Recordar

1. **Recursión**: Las ecuaciones de Bellman expresan valores presentes en términos de valores futuros
2. **Consistencia**: Son condiciones de consistencia que deben satisfacer los valores
3. **Optimalidad**: La ecuación óptima caracteriza completamente V* y Q*
4. **Base de algoritmos**: Casi todos los algoritmos de RL derivan de estas ecuaciones

## Próximos Pasos

Ahora que entiendes las ecuaciones de Bellman, puedes:
1. Estudiar [Value Functions y Políticas](value_policy.md) en profundidad
2. Aprender sobre [Programación Dinámica](../02_algoritmos_clasicos/dynamic_programming/) que resuelve estas ecuaciones
3. Explorar métodos que aproximan estas ecuaciones cuando no conocemos el modelo

## Referencias

- Sutton & Barto, Chapter 3.5-3.8: Bellman Equations
- Bellman, R. (1957). Dynamic Programming
- Bertsekas, D. (2012). Dynamic Programming and Optimal Control
