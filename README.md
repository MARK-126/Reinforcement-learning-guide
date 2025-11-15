# Guía Completa de Reinforcement Learning

Una guía completa y estructurada para aprender Reinforcement Learning (Aprendizaje por Refuerzo) desde los fundamentos hasta implementaciones avanzadas.

## 📚 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Requisitos Previos](#requisitos-previos)
3. [Instalación](#instalación)
4. [Notebooks Interactivos (Ejercicios Fill-in-the-Blank)](#-notebooks-interactivos-ejercicios-fill-in-the-blank)
5. [Estructura del Repositorio](#estructura-del-repositorio)
6. [Ruta de Aprendizaje](#ruta-de-aprendizaje)
7. [Recursos Adicionales](#recursos-adicionales)

## 🎯 Introducción

El Reinforcement Learning es una rama del machine learning donde un agente aprende a tomar decisiones interactuando con un ambiente. A través de prueba y error, el agente recibe recompensas o penalizaciones y aprende a maximizar las recompensas a largo plazo.

Este repositorio está diseñado para proporcionar:
- **Teoría sólida**: Conceptos fundamentales explicados claramente
- **Implementaciones prácticas**: Código funcional de algoritmos clásicos y modernos
- **Ejemplos completos**: Proyectos aplicados a problemas reales
- **Recursos curados**: Referencias a papers, libros y cursos

## 📋 Requisitos Previos

### Conocimientos Recomendados
- **Python**: Programación básica a intermedia
- **Matemáticas**:
  - Álgebra lineal (vectores, matrices)
  - Cálculo (derivadas, optimización)
  - Probabilidad y estadística
- **Machine Learning básico**: Conceptos de supervised learning (opcional pero útil)

### Software Necesario
- Python 3.8 o superior
- pip o conda para gestión de paquetes
- Jupyter Notebook (opcional, para ejemplos interactivos)

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/MARK-126/Reinforcement-learning-guide.git
cd Reinforcement-learning-guide
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar instalación
```bash
python -c "import gym; import torch; print('✓ Instalación exitosa')"
```

## 📓 Notebooks Interactivos (Ejercicios Fill-in-the-Blank)

Este repositorio incluye **notebooks interactivos estilo Coursera/DeepLearning.AI** para los algoritmos principales. Estos notebooks están diseñados con una pedagogía guiada que incluye:

### Características de los Notebooks

✅ **Explicaciones intercaladas**: Teoría mezclada con código
✅ **Ejercicios fill-in-the-blank**: Completa el código en secciones marcadas
✅ **Tests automáticos**: Valida tu implementación con outputs esperados
✅ **Visualizaciones embebidas**: Gráficos para entender el aprendizaje
✅ **Formato guiado**: Paso a paso desde conceptos hasta implementación completa

### Notebooks Disponibles

#### 02_algoritmos_clasicos/temporal_difference/
- **Q_Learning_Exercise.ipynb**: Implementa Q-Learning desde cero
  - Ejercicio 1: Inicialización de Q-table
  - Ejercicio 2: Política epsilon-greedy
  - Ejercicio 3: Regla de actualización Q-Learning
  - Ejercicio 4: Loop de entrenamiento completo
  - Tests: `q_learning_tests.py`

- **SARSA_Exercise.ipynb**: Implementa SARSA y compáralo con Q-Learning
  - Ejercicio 1: Regla de actualización SARSA (on-policy)
  - Ejercicio 2: Loop de entrenamiento SARSA
  - Comparación: CliffWalking environment (SARSA vs Q-Learning)
  - Tests: `sarsa_tests.py`

#### 03_deep_rl/dqn/
- **DQN_Exercise.ipynb**: Deep Q-Network con PyTorch
  - Ejercicio 1: Arquitectura de Q-Network
  - Ejercicio 2: Experience Replay Buffer
  - Ejercicio 3: Cálculo de TD Loss
  - Ejercicio 4: Loop de entrenamiento DQN
  - Tests: `dqn_tests.py`

### Cómo Usar los Notebooks

1. **Abrir el notebook**:
   ```bash
   jupyter notebook 02_algoritmos_clasicos/temporal_difference/Q_Learning_Exercise.ipynb
   ```

2. **Leer las explicaciones**: Cada sección tiene teoría y contexto

3. **Completar los ejercicios**: Busca los comentarios:
   ```python
   # GRADED FUNCTION: nombre_funcion

   def nombre_funcion(params):
       # (approx. 3 lines)
       # Instrucciones claras de qué hacer

       # YOUR CODE STARTS HERE


       # YOUR CODE ENDS HERE

       return resultado
   ```

4. **Ejecutar los tests**: Cada ejercicio tiene una celda de test:
   ```python
   nombre_funcion_test(nombre_funcion)
   ```

5. **Ver el resultado**: Los tests te dirán si tu implementación es correcta

### Estructura de un Ejercicio

```python
# GRADED FUNCTION: q_learning_update

def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
    """
    Update Q-table using Q-Learning rule.

    Arguments:
    Q -- Q-table, numpy array of shape (n_states, n_actions)
    state -- current state
    action -- action taken
    reward -- reward received
    next_state -- next state
    done -- boolean, True if next_state is terminal
    alpha -- learning rate
    gamma -- discount factor

    Returns:
    Q -- updated Q-table
    td_error -- TD error (for tracking)
    """
    # (approx. 5-7 lines)
    # Step 1: Get current Q-value
    # Step 2: Calculate TD target (if done: r, else: r + gamma * max Q(s',a'))
    # Step 3: Calculate TD error
    # Step 4: Update Q-value

    # YOUR CODE STARTS HERE
    current_q = Q[state, action]

    if done:
        target_q = reward
    else:
        target_q = reward + gamma * np.max(Q[next_state])

    td_error = target_q - current_q
    Q[state, action] = current_q + alpha * td_error
    # YOUR CODE ENDS HERE

    return Q, td_error
```

### Tests Automáticos

Cada función tiene tests que verifican:
- ✅ **Tipo de datos**: ¿Devuelve numpy array?
- ✅ **Forma (shape)**: ¿Dimensiones correctas?
- ✅ **Valores**: ¿Cálculos correctos con casos de prueba?

Ejemplo de output de test:
```
Testing q_learning_update...
Non-terminal update:
  Q[0,0] before: 0.0000
  Q[0,0] after: 0.1450
  TD error: 1.4500

Terminal update:
  Q[0,0] after: 0.1000
  TD error: 1.0000

✅ All tests passed!
```

### Ventajas del Formato Interactivo

1. **Aprendizaje Activo**: Escribes código, no solo lo lees
2. **Feedback Inmediato**: Los tests te dicen si vas bien
3. **Progresión Guiada**: De simple a complejo, paso a paso
4. **Validación Rigurosa**: Tests con valores esperados exactos
5. **Visualización**: Gráficos para entender el comportamiento

### Diferencia con Archivos `.py` Existentes

| Aspecto | Archivos `.py` | Notebooks `_Exercise.ipynb` |
|---------|----------------|----------------------------|
| Código | Completo, listo para ejecutar | Ejercicios para completar |
| Uso | Referencia, lectura | Aprendizaje interactivo |
| Tests | No incluidos | Tests automáticos integrados |
| Teoría | Docstrings | Explicaciones markdown extensas |
| Público | Avanzado | Principiantes/Intermedios |

**Recomendación**:
- 📚 **Principiantes**: Empiecen con los notebooks `_Exercise.ipynb`
- 🔧 **Avanzados**: Usen los archivos `.py` como referencia o para proyectos

## 📁 Estructura del Repositorio

```
Reinforcement-learning-guide/
│
├── 01_fundamentos/              # Conceptos básicos de RL
│   ├── introduccion.md          # Qué es RL, historia, aplicaciones
│   ├── mdp.md                   # Procesos de Decisión de Markov
│   ├── bellman.md               # Ecuaciones de Bellman
│   └── value_policy.md          # Value functions y políticas
│
├── 02_algoritmos_clasicos/      # Métodos tabulares y clásicos
│   ├── dynamic_programming/     # Programación dinámica
│   │   ├── policy_iteration.py
│   │   └── value_iteration.py
│   ├── monte_carlo/             # Métodos Monte Carlo
│   │   ├── mc_prediction.py
│   │   └── mc_control.py
│   └── temporal_difference/     # TD Learning
│       ├── sarsa.py
│       ├── q_learning.py
│       └── expected_sarsa.py
│
├── 03_deep_rl/                  # Deep Reinforcement Learning
│   ├── dqn/                     # Deep Q-Networks
│   │   ├── dqn_basic.py
│   │   ├── double_dqn.py
│   │   └── dueling_dqn.py
│   ├── policy_gradient/         # Métodos de gradiente de política
│   │   ├── reinforce.py
│   │   ├── actor_critic.py
│   │   └── a3c.py
│   └── advanced/                # Algoritmos avanzados
│       ├── ppo.py               # Proximal Policy Optimization
│       ├── sac.py               # Soft Actor-Critic
│       └── td3.py               # Twin Delayed DDPG
│
├── 04_ejemplos/                 # Proyectos completos
│   ├── cartpole/                # Balance de péndulo invertido
│   ├── lunar_lander/            # Aterrizaje lunar
│   ├── atari/                   # Juegos de Atari
│   └── custom_env/              # Crear ambientes personalizados
│
├── 05_recursos/                 # Material adicional
│   ├── papers.md                # Papers fundamentales
│   ├── libros.md                # Libros recomendados
│   ├── cursos.md                # Cursos online
│   └── datasets.md              # Datasets y benchmarks
│
├── utils/                       # Utilidades y helpers
│   ├── plotting.py              # Visualización de resultados
│   ├── wrappers.py              # Gym wrappers útiles
│   └── replay_buffer.py         # Experience replay
│
├── requirements.txt             # Dependencias del proyecto
├── .gitignore                   # Archivos a ignorar
└── README.md                    # Este archivo
```

## 🎓 Ruta de Aprendizaje

### Nivel 1: Fundamentos (2-3 semanas)
1. **Semana 1**: Conceptos básicos
   - ¿Qué es RL? Agente, ambiente, recompensa
   - Procesos de Decisión de Markov (MDPs)
   - Value functions y políticas
   - Ecuaciones de Bellman

2. **Semana 2-3**: Métodos tabulares
   - Programación dinámica
   - Métodos Monte Carlo
   - Temporal Difference Learning (TD)
   - Q-Learning y SARSA

### Nivel 2: Deep RL (4-6 semanas)
3. **Semana 4-5**: Deep Q-Learning
   - Redes neuronales básicas con PyTorch
   - DQN (Deep Q-Network)
   - Experience Replay
   - Target Networks
   - Variantes: Double DQN, Dueling DQN

4. **Semana 6-8**: Policy Gradient Methods
   - REINFORCE
   - Actor-Critic
   - Advantage Actor-Critic (A2C/A3C)

### Nivel 3: Avanzado (6-8 semanas)
5. **Semana 9-12**: Algoritmos modernos
   - PPO (Proximal Policy Optimization)
   - DDPG (Deep Deterministic Policy Gradient)
   - TD3 (Twin Delayed DDPG)
   - SAC (Soft Actor-Critic)

6. **Semana 13-16**: Tópicos especiales
   - Multi-agent RL
   - Model-based RL
   - Hierarchical RL
   - Inverse RL

### Proyectos Prácticos Sugeridos
- [ ] Implementar Q-Learning para resolver un GridWorld
- [ ] Entrenar un agente DQN para jugar CartPole
- [ ] Resolver LunarLander con PPO
- [ ] Crear un ambiente custom y entrenar un agente
- [ ] Implementar un algoritmo desde un paper

## 📖 Recursos Adicionales

### Libros Fundamentales
- **"Reinforcement Learning: An Introduction"** - Sutton & Barto (2018)
  - El libro de texto definitivo en RL
  - [Versión gratuita online](http://incompleteideas.net/book/the-book-2nd.html)

- **"Deep Reinforcement Learning Hands-On"** - Maxim Lapan
  - Enfoque práctico con PyTorch
  
- **"Algorithms for Reinforcement Learning"** - Csaba Szepesvári
  - Perspectiva matemática rigurosa

### Cursos Online
- **David Silver's RL Course** (DeepMind)
  - [Videos en YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
  
- **CS285: Deep RL** - UC Berkeley (Sergey Levine)
  - [Página del curso](http://rail.eecs.berkeley.edu/deeprlcourse/)

- **Spinning Up in Deep RL** - OpenAI
  - [Documentación interactiva](https://spinningup.openai.com/)

### Papers Fundamentales
- **DQN**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- **A3C**: Mnih et al. (2016) - "Asynchronous Methods for Deep Reinforcement Learning"
- **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- **SAC**: Haarnoja et al. (2018) - "Soft Actor-Critic"

### Comunidades y Foros
- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/) - Subreddit activo
- [RL Discord](https://discord.gg/xhfNqQv) - Comunidad en Discord
- Stack Overflow con tag `reinforcement-learning`

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Si deseas mejorar este repositorio:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -m 'Añadir nueva explicación'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

## 📝 Notas

- Los ejemplos están diseñados para ejecutarse en CPU, pero muchos se benefician de GPU
- Algunos ambientes (Atari) requieren instalación adicional de ROMs
- Se recomienda usar Google Colab para entrenamientos largos si no tienes GPU local

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**MARK-126**

---

⭐ Si este repositorio te resulta útil, considera darle una estrella!

**¡Feliz aprendizaje de Reinforcement Learning!** 🚀🤖