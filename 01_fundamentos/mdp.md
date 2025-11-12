# Procesos de Decisión de Markov (MDPs)

## ¿Qué es un MDP?

Un **Markov Decision Process (MDP)** es el framework matemático formal para describir ambientes en reinforcement learning. La mayoría de problemas de RL se pueden formalizar como MDPs.

## Definición Formal

Un MDP se define por la tupla (S, A, P, R, γ):

- **S**: Conjunto de estados (State space)
- **A**: Conjunto de acciones (Action space)
- **P**: Función de transición de probabilidad P(s'|s,a)
- **R**: Función de recompensa R(s,a,s')
- **γ**: Factor de descuento (0 ≤ γ ≤ 1)

## La Propiedad de Markov

La característica fundamental de un MDP es la **Propiedad de Markov**:

> "El futuro es independiente del pasado dado el presente"

Matemáticamente:
```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)
```

Esto significa que el estado actual contiene **toda la información necesaria** para predecir el futuro. No necesitamos conocer la historia completa.

### Ejemplo: No Markoviano vs Markoviano

**No Markoviano**: 
- Estado = "Veo una pieza de ajedrez"
- No sabemos dónde están las demás piezas
- Necesitamos historia para decidir

**Markoviano**:
- Estado = "Posición completa del tablero"
- Toda la información necesaria está en el estado actual
- No necesitamos historia

## Componentes Detallados

### 1. Estado (State)

El estado s ∈ S representa la situación actual del ambiente.

**Ejemplos**:
- **GridWorld**: Posición (x, y) del agente
- **CartPole**: (posición, velocidad, ángulo, velocidad angular)
- **Atari**: Píxeles de la pantalla
- **Ajedrez**: Posición de todas las piezas

### 2. Acción (Action)

La acción a ∈ A es lo que el agente puede hacer.

**Tipos de espacios de acción**:

**Discreto finito**:
```python
A = {arriba, abajo, izquierda, derecha}
|A| = 4
```

**Discreto grande**:
```python
A = {mover pieza de X a Y} en ajedrez
|A| ≈ 35 (en promedio)
```

**Continuo**:
```python
A = [-1, 1] × [-1, 1]  # Control 2D
A ⊂ ℝⁿ
```

### 3. Dinámica de Transición

La función de transición P especifica la probabilidad de llegar a un estado s' desde s tomando acción a:

```
P(s'|s,a) = P(S_{t+1} = s' | S_t = s, A_t = a)
```

**Determinista**:
```
P(s'|s,a) = 1 para un s' específico
P(s'|s,a) = 0 para todos los demás
```

**Estocástica**:
```
P(s'|s,a) ∈ [0,1] para varios s'
Σ_s' P(s'|s,a) = 1
```

### 4. Función de Recompensa

La recompensa puede definirse de varias formas:

**R(s, a, s')**: Recompensa por transición específica
```python
R(s="cerca del objetivo", a="avanzar", s'="en el objetivo") = +10
```

**R(s, a)**: Recompensa por estado-acción
```python
R(s="estado peligroso", a="cualquiera") = -1
```

**R(s)**: Recompensa por estado
```python
R(s="estado objetivo") = +1
R(s="otros estados") = 0
```

## Ejemplos de MDPs

### Ejemplo 1: GridWorld Simple

```
Estado: S = {(x,y) | 0 ≤ x,y ≤ 3}  (16 estados)
Acciones: A = {↑, ↓, ←, →}
Transiciones: Deterministas (con paredes)
Recompensas:
  - R(s_goal) = +1
  - R(s_trap) = -1
  - R(otros) = -0.01 (costo por paso)
γ = 0.9
```

### Ejemplo 2: Inventario

```
Estado: s = cantidad en inventario (0 a 10)
Acciones: a = cuánto ordenar (0 a 5)
Transiciones: Probabilísticas (demanda aleatoria)
Recompensas:
  - Ventas: +10 por unidad
  - Costo de orden: -2 por unidad
  - Costo de almacén: -1 por unidad almacenada
```

### Ejemplo 3: CartPole

```
Estado: s = (x, ẋ, θ, θ̇)  # Posición, velocidad, ángulo, velocidad angular
  - x ∈ [-2.4, 2.4]
  - θ ∈ [-12°, 12°]
Acciones: A = {izquierda, derecha}
Transiciones: Física simulada
Recompensas:
  - R = +1 por cada step que no cae
  - R = 0 cuando termina (cae o sale de rango)
```

## Episodios y Horizontes

### Tareas Episódicas
- Tienen un **estado terminal** claro
- Duración finita T
- Retorno: G_t = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}

**Ejemplos**: Juegos (ajedrez, Go), episodios de CartPole

### Tareas Continuas
- No tienen estado terminal natural
- Duración potencialmente infinita
- Retorno: G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}

**Ejemplos**: Control de procesos, trading, robots autónomos

## Política (Policy)

Una política π mapea estados a acciones:

**Determinista**:
```
π: S → A
π(s) = a
```

**Estocástica**:
```
π: S × A → [0,1]
π(a|s) = P(A_t = a | S_t = s)
```

## Funciones de Valor

### State-Value Function
Valor esperado de estar en estado s siguiendo política π:

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
```

### Action-Value Function
Valor esperado de tomar acción a en estado s y seguir π:

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]
```

## Política Óptima

La **política óptima** π* maximiza el valor en todos los estados:

```
π* = argmax_π V^π(s)  para todo s ∈ S
```

Propiedades:
- Siempre existe al menos una política óptima
- Todas las políticas óptimas comparten las mismas funciones de valor óptimas V* y Q*
- Una política óptima puede ser determinista

## Partially Observable MDPs (POMDPs)

En algunos casos, el agente no observa el estado completo, sino una **observación** o:

```
POMDP = (S, A, P, R, Ω, O, γ)
```

Donde:
- **Ω**: Conjunto de observaciones
- **O**: Función de observación O(o|s,a)

**Ejemplo**: Poker (no vemos cartas del oponente)

## Resumen de Propiedades

| Propiedad | Descripción | Impacto |
|-----------|-------------|---------|
| Markoviana | Futuro independiente del pasado dado presente | Simplifica algoritmos |
| Estocástica | Transiciones probabilísticas | Mayor complejidad |
| Episódica | Tiene estado terminal | Simplifica retorno |
| Observable | Estado completamente visible | MDP vs POMDP |

## Implementación en Código

```python
# Ejemplo simple de MDP en Python
class SimpleMDP:
    def __init__(self):
        self.states = [...]
        self.actions = [...]
        self.gamma = 0.9
    
    def transition(self, state, action):
        """Retorna (next_state, reward, done)"""
        pass
    
    def get_reward(self, state, action, next_state):
        """Calcula recompensa"""
        pass
```

## Ejercicios

1. **Diseña un MDP** para el problema de encontrar el camino más corto en un laberinto
2. **Verifica la propiedad de Markov**: ¿Es el estado actual de un juego de Blackjack Markoviano?
3. **Calcula el retorno**: Dada la secuencia de recompensas [1, -1, 1, -1, 10] y γ=0.9, ¿cuál es G_0?

## Próximos Pasos

Con el entendimiento de MDPs, ahora puedes estudiar:
1. [Ecuaciones de Bellman](bellman.md) - Relaciones recursivas de valor
2. [Value Functions y Políticas](value_policy.md) - Profundización en funciones de valor

## Referencias

- Sutton & Barto, Chapter 3: Finite Markov Decision Processes
- Bellman, R. (1957). A Markovian Decision Process
- Puterman, M. L. (2014). Markov Decision Processes
