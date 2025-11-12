# Introducción al Reinforcement Learning

## ¿Qué es el Reinforcement Learning?

El **Reinforcement Learning (RL)** o Aprendizaje por Refuerzo es un paradigma del machine learning donde un agente aprende a tomar decisiones mediante la interacción con un ambiente. A diferencia del aprendizaje supervisado (donde se tienen etiquetas correctas) o no supervisado (donde se buscan patrones), en RL el agente aprende mediante **prueba y error**, recibiendo recompensas o penalizaciones por sus acciones.

## Componentes Fundamentales

### 1. Agente (Agent)
El **agente** es quien toma las decisiones. Es el "cerebro" que aprende a actuar de manera óptima.

### 2. Ambiente (Environment)
El **ambiente** es el mundo con el que el agente interactúa. Responde a las acciones del agente cambiando de estado y proporcionando recompensas.

### 3. Estado (State)
El **estado** es una representación de la situación actual del ambiente. Puede ser:
- **Completamente observable**: El agente ve todo el estado
- **Parcialmente observable**: El agente solo ve parte del estado

### 4. Acción (Action)
Una **acción** es lo que el agente puede hacer en cada estado. Las acciones pueden ser:
- **Discretas**: Un conjunto finito (ej: arriba, abajo, izquierda, derecha)
- **Continuas**: Valores en un rango (ej: velocidad de 0 a 100)

### 5. Recompensa (Reward)
La **recompensa** es una señal numérica que el ambiente le da al agente después de cada acción. Indica qué tan buena fue esa acción.

### 6. Política (Policy)
La **política** (π) es la estrategia del agente: un mapeo de estados a acciones.
- **Determinista**: π(s) → a (una acción específica por estado)
- **Estocástica**: π(a|s) → probabilidad (distribución de probabilidad sobre acciones)

## El Ciclo de Interacción

```
1. Agente observa estado s_t
2. Agente selecciona acción a_t según su política π
3. Ambiente recibe a_t y transiciona a nuevo estado s_{t+1}
4. Ambiente proporciona recompensa r_{t+1}
5. Agente actualiza su conocimiento
6. Repetir...
```

## Objetivo del RL

El objetivo es aprender una **política óptima** π* que maximice el **retorno esperado** (suma de recompensas futuras):

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ γ^k R_{t+k+1}
```

Donde:
- **G_t**: Retorno desde el tiempo t
- **γ** (gamma): Factor de descuento (0 ≤ γ ≤ 1)
- **R_t**: Recompensa en el tiempo t

## Factor de Descuento (γ)

El factor de descuento determina la importancia de recompensas futuras:
- **γ = 0**: Agente miope, solo considera recompensa inmediata
- **γ = 1**: Agente prevé infinitamente al futuro
- **0 < γ < 1**: Balance entre recompensas inmediatas y futuras (típicamente 0.9-0.99)

## Historia del Reinforcement Learning

### Orígenes (1950s-1980s)
- **1954**: Bellman desarrolla programación dinámica
- **1989**: Watkins introduce Q-Learning

### Era Moderna (1990s-2000s)
- **1992**: TD-Gammon (Tesauro) - Backgammon nivel mundial
- **1998**: Sutton & Barto publican primera edición de su libro

### Deep RL Era (2010s-Presente)
- **2013**: DQN juega Atari (DeepMind)
- **2015**: DQN paper publicado en Nature
- **2016**: AlphaGo vence a Lee Sedol
- **2017**: PPO se convierte en algoritmo estándar
- **2019**: OpenAI Five juega Dota 2
- **2020+**: Avances en RL aplicado (robótica, conducción autónoma)

## Aplicaciones del RL

### Juegos
- Video juegos (Atari, StarCraft, Dota)
- Juegos de mesa (Go, Ajedrez)

### Robótica
- Manipulación de objetos
- Locomoción
- Control de drones

### Sistemas Autónomos
- Conducción autónoma
- Control de tráfico

### Finanzas
- Trading algorítmico
- Gestión de portafolios

### Salud
- Tratamientos personalizados
- Diseño de medicamentos

### Sistemas de Recomendación
- Publicidad online
- Contenido personalizado

### Optimización de Recursos
- Centros de datos (enfriamiento)
- Redes energéticas

## Tipos de RL

### Por tipo de aprendizaje:
1. **Model-Free**: No aprende modelo del ambiente
   - Value-based (Q-Learning, DQN)
   - Policy-based (REINFORCE, PPO)
   - Actor-Critic (A3C, SAC)

2. **Model-Based**: Aprende modelo del ambiente
   - Dyna-Q
   - AlphaZero

### Por tipo de política:
1. **On-Policy**: Aprende sobre la política que está ejecutando (SARSA, PPO)
2. **Off-Policy**: Aprende sobre política diferente a la que ejecuta (Q-Learning, DQN)

## Desafíos en RL

1. **Exploración vs Explotación**: Balance entre probar nuevas acciones y usar conocimiento actual
2. **Sparse Rewards**: Recompensas infrecuentes dificultan el aprendizaje
3. **Credit Assignment**: ¿Qué acción causó la recompensa?
4. **Sample Efficiency**: RL típicamente requiere muchas muestras
5. **Estabilidad**: Entrenamiento puede ser inestable

## Comparación con otros paradigmas de ML

| Aspecto | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|---------|-------------------|---------------------|---------------------|
| Datos | Etiquetados | Sin etiquetas | Señal de recompensa |
| Feedback | Explícito | Ninguno | Evaluativo (bueno/malo) |
| Objetivo | Predecir etiquetas | Encontrar estructura | Maximizar recompensas |
| Ejemplo | Clasificación de imágenes | Clustering | Jugar ajedrez |

## Próximos Pasos

Ahora que entiendes los conceptos básicos, continúa con:
1. [Procesos de Decisión de Markov (MDPs)](mdp.md)
2. [Ecuaciones de Bellman](bellman.md)
3. [Value Functions y Políticas](value_policy.md)

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.)
- Silver, D. (2015). UCL Course on RL - Lecture 1
