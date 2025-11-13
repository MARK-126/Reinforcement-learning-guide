# DQN y sus Variantes

Este directorio contiene implementaciones de Deep Q-Networks (DQN) y sus variantes más importantes para Deep Reinforcement Learning.

## 📁 Archivos Implementados

### 1. `dqn_basic.py` - DQN Básico
Implementación del DQN original (Mnih et al., 2015).

**Características:**
- Q-Network con arquitectura MLP
- Experience Replay Buffer
- Target Network con actualizaciones periódicas
- Exploración ε-greedy con decaimiento
- Ejemplo con CartPole-v1

**Componentes:**
- `DQN`: Red neuronal para Q-values
- `ReplayBuffer`: Buffer de experiencias
- `DQNAgent`: Agente con algoritmo DQN
- `train_dqn()`: Función de entrenamiento
- `plot_training()`: Visualización de resultados

### 2. `double_dqn.py` - Double DQN ⭐ NUEVO
Implementación de Double DQN (van Hasselt et al., 2015).

**Mejora clave:** Reduce la sobreestimación de Q-values al separar la selección y evaluación de acciones.

**Diferencia con DQN estándar:**
```python
# DQN estándar
target = r + γ * max_a' Q_target(s', a')

# Double DQN
a* = argmax_a' Q_online(s', a')  # Seleccionar con online network
target = r + γ * Q_target(s', a*)  # Evaluar con target network
```

**Características:**
- Arquitectura idéntica a DQN básico
- Double Q-learning update
- Soporte para hard y soft updates
- Compatibilidad con tau para actualización suave
- Ejemplos con CartPole y LunarLander

**Componentes:**
- `DQN`: Red neuronal (misma que básico)
- `ReplayBuffer`: Buffer de experiencias
- `DoubleDQNAgent`: Agente con Double DQN
- `train_double_dqn()`: Entrenamiento
- `evaluate_agent()`: Evaluación sin exploración
- `plot_training()`: Visualización
- `compare_with_standard_dqn()`: Comparación con DQN estándar

**Cuándo usar:**
- Cuando DQN estándar sobreestima Q-values
- En ambientes con recompensas ruidosas
- Para aprendizaje más estable
- Mismo costo computacional que DQN estándar

### 3. `dueling_dqn.py` - Dueling DQN ⭐ NUEVO
Implementación de Dueling DQN (Wang et al., 2016).

**Mejora clave:** Separa la estimación del valor del estado de las ventajas de las acciones.

**Arquitectura:**
```
Estado → Features (compartidas)
         ↓
         ├→ Value Stream → V(s)
         └→ Advantage Stream → A(s,a)

Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
```

**Características:**
- Arquitectura Dueling con value y advantage streams
- Se puede combinar con Double DQN (recomendado)
- Método `analyze_value_advantage()` para inspección
- Soporte para hard y soft updates
- Ejemplos con CartPole y LunarLander

**Componentes:**
- `DuelingDQN`: Red neuronal con arquitectura Dueling
- `ReplayBuffer`: Buffer de experiencias
- `DuelingDQNAgent`: Agente con opción de Double DQN
- `train_dueling_dqn()`: Entrenamiento
- `evaluate_agent()`: Evaluación con análisis opcional
- `plot_training()`: Visualización de entrenamiento
- `visualize_value_advantage()`: Análisis de V(s) y A(s,a)

**Cuándo usar:**
- Cuando el valor del estado es importante independiente de la acción
- En espacios de acción grandes
- Cuando muchas acciones tienen efectos similares
- Para mejor generalización
- Combinar con Double DQN para mejores resultados

## 🚀 Uso Rápido

### Ejecutar ejemplos individuales

```bash
# DQN básico en CartPole
cd 03_deep_rl/dqn
python dqn_basic.py

# Double DQN en LunarLander
python double_dqn.py

# Dueling DQN en LunarLander
python dueling_dqn.py
```

### Usar como librería

```python
# Importar desde el paquete
from dqn import DoubleDQNAgent, DuelingDQNAgent

# O importar directamente
from dqn.double_dqn import DoubleDQNAgent
from dqn.dueling_dqn import DuelingDQNAgent

# Crear ambiente
import gymnasium as gym
env = gym.make('LunarLander-v2')

# Double DQN
agent = DoubleDQNAgent(
    state_dim=8,
    action_dim=4,
    learning_rate=5e-4,
    gamma=0.99,
    epsilon_decay=1000,
    tau=0.005  # Soft update
)

# Dueling DQN + Double DQN
agent = DuelingDQNAgent(
    state_dim=8,
    action_dim=4,
    learning_rate=5e-4,
    gamma=0.99,
    use_double_dqn=True,  # Combinar con Double DQN
    tau=0.005
)

# Entrenar
from dqn.double_dqn import train_double_dqn
rewards, losses = train_double_dqn(env, agent, n_episodes=500)
```

## 📊 Comparación de Variantes

| Característica | DQN Básico | Double DQN | Dueling DQN |
|----------------|------------|------------|-------------|
| **Arquitectura** | MLP simple | MLP simple | Value + Advantage streams |
| **Update** | max Q_target | Decouple select/eval | Igual que Double |
| **Parámetros** | ~10K | ~10K | ~12K |
| **Problema que resuelve** | Q-learning + DL | Sobreestimación | Generalización |
| **Costo computacional** | Base | Igual | +20% |
| **Mejora típica** | Base | +10-15% | +15-25% |
| **Combinable** | - | Con Dueling | Con Double |

## 🔧 Características Comunes

Todas las implementaciones incluyen:

### Componentes
- ✅ PyTorch neural networks
- ✅ Experience replay buffer
- ✅ ε-greedy exploration con decaimiento exponencial
- ✅ Target network con hard/soft updates
- ✅ Type hints completos
- ✅ Docstrings en español

### Funcionalidades
- ✅ Training loops completos
- ✅ Evaluación sin exploración
- ✅ Visualización y tracking de métricas
- ✅ Save/load de modelos
- ✅ `__main__` con ejemplos completos
- ✅ Logging detallado

### Hiperparámetros Configurables
- `learning_rate`: Tasa de aprendizaje (1e-4 a 1e-3)
- `gamma`: Factor de descuento (0.95 a 0.99)
- `epsilon_start/end/decay`: Exploración
- `buffer_size`: Tamaño del replay buffer
- `batch_size`: Tamaño del mini-batch
- `target_update`: Frecuencia de actualización (episodios)
- `tau`: Factor de soft update (None = hard update)
- `hidden_dim`: Dimensión de capas ocultas

## 🎯 Configuraciones Recomendadas

### CartPole-v1 (simple)
```python
config = {
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'epsilon_decay': 500,
    'buffer_size': 10000,
    'batch_size': 64,
    'target_update': 10,
    'tau': None,  # Hard update
    'hidden_dim': 128,
    'n_episodes': 300
}
```

### LunarLander-v2 (complejo)
```python
config = {
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'epsilon_decay': 1000,
    'buffer_size': 50000,
    'batch_size': 128,
    'target_update': 5,
    'tau': 0.005,  # Soft update
    'hidden_dim': 256,
    'n_episodes': 600
}
```

## 📈 Resultados Esperados

### CartPole-v1
- **DQN Básico**: ~200 reward en 200 episodios
- **Double DQN**: ~250 reward en 150 episodios
- **Dueling DQN**: ~300 reward en 100 episodios

### LunarLander-v2
- **DQN Básico**: ~150 reward en 500 episodios
- **Double DQN**: ~180 reward en 450 episodios
- **Dueling DQN + Double**: ~200 reward en 400 episodios

## 🧪 Testing

Ejecutar el script de verificación:

```bash
# Requiere: torch, gymnasium, numpy, matplotlib
python test_dqn_variants.py
```

Tests incluidos:
- ✓ Arquitecturas de redes
- ✓ Agentes y métodos
- ✓ Training loops
- ✓ Save/load de modelos
- ✓ Comparación de arquitecturas

## 📚 Referencias

1. **DQN**: Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.

2. **Double DQN**: van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.

3. **Dueling DQN**: Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML.

## 🎓 Conceptos Educativos

### Experience Replay
Almacena transiciones (s, a, r, s', done) y muestrea mini-batches aleatorios para:
- Romper correlación temporal
- Reutilizar experiencias
- Estabilizar entrenamiento

### Target Network
Red separada para calcular targets, actualizada periódicamente:
- **Hard update**: Copia completa cada N episodios
- **Soft update**: Actualización gradual con τ cada step

### ε-greedy Exploration
Balance exploración/explotación:
- ε alto al inicio → explorar
- ε bajo al final → explotar
- Decaimiento exponencial

### Double Q-learning
Reduce sobreestimación:
- Seleccionar: argmax sobre online network
- Evaluar: Q-value de target network

### Dueling Architecture
Separa valor y ventajas:
- V(s): Cuán bueno es el estado
- A(s,a): Ventaja relativa de cada acción
- Q(s,a) = V(s) + (A(s,a) - mean(A))

## 🔄 Próximos Pasos

Para continuar aprendiendo Deep RL:

1. **Mejoras a DQN**:
   - Prioritized Experience Replay
   - Noisy Networks
   - Distributional RL (C51, QR-DQN)
   - Rainbow (combina todas las mejoras)

2. **Policy Gradient Methods**:
   - REINFORCE
   - Actor-Critic
   - A3C/A2C
   - PPO

3. **Continuous Control**:
   - DDPG
   - TD3
   - SAC

4. **Meta-Learning**:
   - MAML
   - Reptile
   - Model-Agnostic Meta-Learning

## 💡 Tips de Uso

1. **Empezar simple**: Probar primero con CartPole antes de LunarLander
2. **Monitorear epsilon**: Asegurar que decae apropiadamente
3. **Tamaño del buffer**: Más grande es mejor, pero usa más memoria
4. **Learning rate**: Si diverge, reducir; si aprende lento, aumentar
5. **Combinar técnicas**: Dueling + Double DQN suele funcionar mejor
6. **Guardar modelos**: Usar `save_every` para checkpoints regulares
7. **Visualizar**: Las gráficas ayudan a detectar problemas

## ⚙️ Dependencias

```bash
pip install torch gymnasium numpy matplotlib
```

Versiones recomendadas:
- Python >= 3.8
- PyTorch >= 2.0
- Gymnasium >= 0.28
- NumPy >= 1.20
- Matplotlib >= 3.5

---

**Nota**: Estas implementaciones están diseñadas para ser educativas y entender los fundamentos de Deep RL antes de pasar a meta-learning. El código está bien documentado en español con comentarios explicativos.
