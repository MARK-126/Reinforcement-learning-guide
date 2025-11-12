# Guía de Inicio Rápido

¡Bienvenido! Esta guía te ayudará a empezar con Reinforcement Learning en 30 minutos.

## 🚀 Configuración Rápida (5 minutos)

### 1. Clonar el Repositorio

```bash
git clone https://github.com/MARK-126/Reinforcement-learning-guide.git
cd Reinforcement-learning-guide
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno
python -m venv venv

# Activar (Linux/Mac)
source venv/bin/activate

# Activar (Windows)
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- `gymnasium` - Ambientes de RL
- `torch` - Deep Learning
- `numpy` - Computación numérica
- `matplotlib` - Visualización
- Y más...

### 4. Verificar Instalación

```bash
python -c "import gymnasium; import torch; print('✓ Listo para empezar!')"
```

## 📖 Tu Primera Sesión de RL (25 minutos)

### Parte 1: Entender los Conceptos (10 minutos)

Lee estos archivos en orden:

1. **[Introducción al RL](01_fundamentos/introduccion.md)** (5 min)
   - ¿Qué es RL?
   - Agente, ambiente, recompensa
   - Exploración vs Explotación

2. **[MDPs Básicos](01_fundamentos/mdp.md)** (5 min - solo introducción)
   - Estados y acciones
   - La propiedad de Markov

### Parte 2: Ejecutar tu Primer Agente (10 minutos)

```bash
# Navegar al ejemplo de CartPole
cd 04_ejemplos/cartpole

# Ejecutar Q-Learning
python cartpole_qlearning.py
```

Esto:
1. ✅ Entrenará un agente por 500 episodios (~2 minutos)
2. ✅ Mostrará progreso en consola
3. ✅ Evaluará el agente entrenado
4. ✅ Generará gráficos de resultados

**Observa**:
- Cómo la recompensa mejora con el tiempo
- Cómo epsilon decae (menos exploración)
- El agente aprende a balancear el poste

### Parte 3: Entender el Código (5 minutos)

Abre `cartpole_qlearning.py` y observa:

```python
# 1. Crear agente
agent = QLearningAgent(
    n_actions=2,        # Izquierda o derecha
    alpha=0.1,          # Learning rate
    gamma=0.99,         # Discount factor
    epsilon=1.0         # Exploration rate
)

# 2. Entrenar
for episode in range(n_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        # Seleccionar acción
        action = agent.get_action(state)
        
        # Ejecutar acción
        next_state, reward, done = env.step(action)
        
        # Actualizar Q-values (¡AQUÍ ESTÁ LA MAGIA!)
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        if done:
            break
```

**Conceptos clave**:
- **Q-Learning**: Aprende valor de estado-acción
- **ε-greedy**: Balance exploración/explotación
- **Update**: Ecuación de Bellman para mejorar estimaciones

## 🎯 Próximos Pasos

### Opción A: Profundizar en Teoría (Recomendado)

1. Lee [MDPs](01_fundamentos/mdp.md) completo
2. Lee [Ecuaciones de Bellman](01_fundamentos/bellman.md)
3. Lee [Value Functions](01_fundamentos/value_policy.md)

**Tiempo**: 2-3 horas  
**Resultado**: Entenderás la base matemática

### Opción B: Más Práctica

1. Modifica `cartpole_qlearning.py`:
   ```python
   # Experimenta cambiando:
   alpha = 0.5        # ¿Más rápido?
   gamma = 0.95       # ¿Diferente?
   epsilon_decay = 0.99  # ¿Más exploración?
   ```

2. Prueba otros ambientes:
   ```python
   env = gym.make('FrozenLake-v1')  # Más simple
   env = gym.make('MountainCar-v0')  # Más difícil
   ```

3. Implementa SARSA:
   - Ve a `02_algoritmos_clasicos/temporal_difference/sarsa.py`
   - Compara con Q-Learning

**Tiempo**: 2-3 horas  
**Resultado**: Intuición práctica de RL

### Opción C: Deep RL

1. Ejecuta DQN en CartPole:
   ```bash
   cd 03_deep_rl/dqn
   python dqn_basic.py
   ```

2. Compara con Q-Learning:
   - ¿Cuál es más rápido?
   - ¿Cuál obtiene mejor resultado?
   - ¿Por qué?

**Tiempo**: 1-2 horas  
**Resultado**: Introducción a Deep RL

## 🛠️ Herramientas Útiles

### Visualizar Agente Entrenado

```python
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
# ... entrenar agente ...

# Visualizar
for episode in range(5):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

env.close()
```

### Debuggear Q-Values

```python
# Ver Q-values para un estado
state = (0, 0, 0, 0)  # Estado ejemplo
print(f"Q-values: {agent.Q[state]}")
print(f"Mejor acción: {np.argmax(agent.Q[state])}")
```

### Guardar/Cargar Agente

```python
import pickle

# Guardar
with open('agent.pkl', 'wb') as f:
    pickle.dump(agent.Q, f)

# Cargar
with open('agent.pkl', 'rb') as f:
    agent.Q = pickle.load(f)
```

## 📚 Recursos de Referencia Rápida

### Ecuaciones Importantes

**Q-Learning Update**:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**SARSA Update**:
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

**ε-greedy**:
```python
if random() < epsilon:
    return random_action()
else:
    return argmax(Q[state])
```

### Hiperparámetros Típicos

| Parámetro | Símbolo | Valor Típico | Descripción |
|-----------|---------|--------------|-------------|
| Learning rate | α | 0.1 - 0.01 | Qué tan rápido aprende |
| Discount factor | γ | 0.99 - 0.95 | Importancia del futuro |
| Epsilon start | ε₀ | 1.0 | Exploración inicial |
| Epsilon end | ε_min | 0.01 | Exploración final |
| Epsilon decay | - | 0.995 | Velocidad de decaimiento |

## ❓ Troubleshooting

### "No module named 'gymnasium'"

```bash
pip install gymnasium
```

### El agente no aprende

1. **Verifica hiperparámetros**: α muy bajo o γ muy alto
2. **Exploración**: Asegúrate que ε esté decayendo
3. **Recompensas**: Verifica que las recompensas tengan sentido
4. **Episodios**: Tal vez necesitas más episodios

### Entrenamiento muy lento

1. **Reduce episodios**: Empieza con 100-200
2. **Discretización**: Si usas Q-Learning, reduce bins
3. **Max steps**: Limita steps por episodio

### Resultados inconsistentes

Esto es normal en RL! Para resultados más estables:

```python
# Fija random seed
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

## 🎓 Plan de 7 Días

### Día 1: Setup y Primer Agente
- ✅ Configurar entorno
- ✅ Ejecutar CartPole
- ✅ Entender el código

### Día 2: Teoría Fundamental
- 📖 Leer fundamentos completos
- 📖 Hacer ejercicios mentales

### Día 3: Q-Learning Profundo
- 💻 Implementar Q-Learning desde cero
- 💻 Probar en FrozenLake

### Día 4: Comparar Algoritmos
- 💻 SARSA vs Q-Learning
- 📊 Comparar resultados

### Día 5: Intro a Deep RL
- 💻 Ejecutar DQN
- 📖 Entender diferencias

### Día 6: Proyecto Personal
- 🚀 Resolver un ambiente nuevo
- 🚀 Experimentar con hiperparámetros

### Día 7: Documentar y Compartir
- 📝 Escribir sobre lo aprendido
- 🌟 Compartir resultados

## 🌟 Consejos de Expertos

1. **Empieza simple**: No saltes directo a Deep RL
2. **Visualiza**: Grafica todo (rewards, Q-values, políticas)
3. **Experimenta**: Cambia hiperparámetros y observa
4. **Lee código**: Las implementaciones son mejores maestros que teoría sola
5. **Sé paciente**: RL es difícil, los agentes no siempre aprenden a la primera

## 📞 Ayuda

¿Atascado? 
- 🐛 [Reporta un issue](https://github.com/MARK-126/Reinforcement-learning-guide/issues)
- 💬 Consulta [CONTRIBUTING.md](CONTRIBUTING.md)
- 📚 Lee la [documentación completa](README.md)

---

**¡Feliz aprendizaje! 🚀🤖**

La mejor forma de aprender RL es implementando. No te preocupes si no entiendes todo al principio, ¡eso es completamente normal!
