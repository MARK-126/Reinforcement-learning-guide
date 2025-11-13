# 🚀 Reinforcement Learning: Guía Profesional Completa

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-100%2B-brightgreen.svg)]()

Una guía completa y profesional para dominar Reinforcement Learning desde los fundamentos matemáticos hasta algoritmos state-of-the-art (SOTA). Incluye **6 notebooks interactivos en formato DeepLearning.AI** con teoría rigurosa, implementaciones desde cero y más de 100 tests automatizados.

---

## 📚 Tabla de Contenidos

- [✨ Características Principales](#-características-principales)
- [🎯 Para Quién es Este Repositorio](#-para-quién-es-este-repositorio)
- [📊 Estadísticas del Proyecto](#-estadísticas-del-proyecto)
- [🗂️ Notebooks Interactivos](#️-notebooks-interactivos)
- [✅ Algoritmos Implementados](#-algoritmos-implementados)
- [🚀 Inicio Rápido](#-inicio-rápido)
- [📁 Estructura del Repositorio](#-estructura-del-repositorio)
- [🎓 Ruta de Aprendizaje](#-ruta-de-aprendizaje)
- [🔬 Referencias Académicas](#-referencias-académicas)
- [📖 Recursos Adicionales](#-recursos-adicionales)
- [🤝 Contribuciones](#-contribuciones)
- [📄 Licencia](#-licencia)

---

## ✨ Características Principales

### 🎓 Notebooks Profesionales Estilo DeepLearning.AI

Los 6 notebooks incluidos siguen el formato profesional de cursos de DeepLearning.AI:

✅ **Table of Contents** con anchor links navegables
✅ **29 ejercicios formales** con scaffolding (`YOUR CODE STARTS/ENDS HERE`)
✅ **100+ tests automatizados** integrados en los notebooks
✅ **"What you should remember"** boxes con puntos clave
✅ **15+ tablas HTML** de comparación de algoritmos
✅ **50+ ecuaciones LaTeX** con explicaciones detalladas
✅ **Visualizaciones interactivas** con matplotlib/seaborn
✅ **Referencias a papers** específicos (Mnih, Schulman, Haarnoja, etc.)
✅ **Código de producción** con type hints y documentación completa

### 🛠️ Implementaciones Completas

- **15+ algoritmos RL** desde cero en PyTorch
- **20,000+ líneas de código** de calidad profesional
- **6 módulos utils** con funciones auxiliares y tests
- **Arquitecturas completas** (DQN, Actor-Critic, Dueling Networks)
- **Ambientes de prueba** (CartPole, LunarLander, GridWorld)

### 🧪 Testing Riguroso

- **100+ test cases** automatizados
- **Coverage completo** de todas las funcionalidades
- **GitHub Actions CI/CD** para validación continua
- **Tests integrados** en notebooks para aprendizaje interactivo

---

## 🎯 Para Quién es Este Repositorio

### ✅ Ideal Para:

- **Estudiantes** que quieren dominar RL con fundamentos sólidos
- **Investigadores** que necesitan implementaciones de referencia
- **Profesionales** preparándose para ML/AI roles
- **Autodidactas** buscando material estructurado de calidad
- **Preparación para Meta-Learning** con bases rigurosas en RL

### 📋 Requisitos Previos

**Conocimientos:**
- Python intermedio (clases, decoradores, tipo hints)
- Matemáticas: Álgebra lineal, Cálculo, Probabilidad
- (Opcional) PyTorch básico - se enseña en los notebooks

**Software:**
- Python 3.8+
- 4GB RAM (8GB recomendado para entrenamientos)
- GPU opcional pero recomendada para Deep RL

---

## 📊 Estadísticas del Proyecto

| Métrica | Cantidad |
|---------|----------|
| **Notebooks Tutoriales** | 6 (260 KB) |
| **Ejercicios Formales** | 29 |
| **Tests Automatizados** | 100+ |
| **Algoritmos Implementados** | 15+ |
| **Líneas de Código** | 20,000+ |
| **Módulos Utils** | 6 (87 KB) |
| **Papers Citados** | 10+ |
| **Tablas de Comparación** | 15+ |
| **Ecuaciones LaTeX** | 50+ |
| **Ambientes de Ejemplo** | 5 |

---

## 🗂️ Notebooks Interactivos

Todos los notebooks siguen el formato profesional de **DeepLearning.AI** con estructura pedagógica optimizada:

### 📘 01. Dynamic Programming (47 KB)

**Contenido:**
- Teoría de MDPs y ecuaciones de Bellman
- Policy Iteration y Value Iteration
- **5 ejercicios** con scaffolding completo
- **6 test suites** integrados
- Visualización de políticas y valores
- Experimentos con diferentes parámetros

**Ejercicios:**
1. `policy_evaluation` - Evaluar políticas con Bellman
2. `policy_improvement` - Mejora greedy de políticas
3. `value_iteration_step` - Un paso de iteración de valor
4. `extract_policy` - Extraer política desde valores
5. `compare_algorithms` - Comparación PI vs VI

**Referencias:** Sutton & Barto (2018), Bellman (1957), Puterman (1994)

---

### 📗 02. Monte Carlo Methods (31 KB)

**Contenido:**
- Aprendizaje model-free desde experiencia
- First-Visit vs Every-Visit MC
- On-Policy vs Off-Policy control
- Importance Sampling
- **5 ejercicios** implementando MC desde cero
- **5 test suites** con validación completa

**Ejercicios:**
1. `implement_first_visit_mc_prediction`
2. `implement_every_visit_mc_prediction`
3. `implement_mc_control_on_policy` (ε-greedy)
4. `implement_mc_control_off_policy`
5. `implement_importance_sampling`

**Tablas de Comparación:**
- First-Visit vs Every-Visit
- MC vs Dynamic Programming
- On-Policy vs Off-Policy

---

### 📙 03. Temporal Difference Learning (51 KB)

**Contenido:**
- Q-Learning (off-policy)
- SARSA (on-policy)
- Expected SARSA
- TD(λ) con eligibility traces
- **4 ejercicios** principales
- **20+ test cases** integrados

**Ejercicios:**
1. `implement_q_learning`
2. `implement_sarsa`
3. `implement_expected_sarsa`
4. `compare_td_methods`

**Ambientes:** FrozenLake, CliffWalking

---

### 📕 04. Deep Q-Networks (44 KB)

**Contenido:**
- Introducción a Deep RL
- DQN con Experience Replay
- Double DQN (reduce overestimation)
- Dueling DQN (arquitectura Value-Advantage)
- **5 ejercicios** con PyTorch
- **21 tests** de arquitecturas y training

**Ejercicios:**
1. `implement_dqn_network` - Red neuronal básica
2. `implement_replay_buffer` - Experience replay
3. `implement_dqn_update` - Loop de entrenamiento
4. `implement_double_dqn` - Desacoplamiento acción-evaluación
5. `implement_dueling_dqn` - Streams V y A

**Papers:**
- Mnih et al. (2015) - DQN original
- van Hasselt et al. (2015) - Double DQN
- Wang et al. (2016) - Dueling DQN

**Tabla de Comparación:** DQN vs Double DQN vs Dueling DQN (6 aspectos)

---

### 📔 05. Policy Gradient Methods (42 KB)

**Contenido:**
- Policy Gradient Theorem (derivación completa)
- REINFORCE con baseline
- Actor-Critic (A2C)
- Generalized Advantage Estimation (GAE)
- **5 ejercicios** implementando PG
- **5 test suites** completas

**Ejercicios:**
1. `implement_policy_network` (discrete + continuous)
2. `implement_reinforce_loss`
3. `implement_baseline` (Value Network)
4. `implement_actor_critic` (A2C con GAE)
5. `implement_gae` (cálculo de advantages)

**Ecuaciones:**
- Policy Gradient Theorem
- REINFORCE formula
- Baseline optimal
- GAE recursion

---

### 📓 06. Advanced SOTA Algorithms (45 KB)

**Contenido:**
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic con auto-tuning)
- **5 ejercicios** implementando SOTA
- **6 test classes** profesionales

**Ejercicios:**
1. `implement_ppo_loss` - Clipped objective
2. `implement_ddpg_networks` - Actor-Critic arquitectura
3. `implement_td3_updates` - Twin Q + delayed updates
4. `implement_sac_temperature` - Auto-tuning de α
5. `compare_sota_algorithms` - Análisis comparativo

**Papers con Links ArXiv:**
- Schulman et al. (2017) - PPO - https://arxiv.org/abs/1707.06347
- Lillicrap et al. (2016) - DDPG - https://arxiv.org/abs/1509.02971
- Fujimoto et al. (2018) - TD3 - https://arxiv.org/abs/1802.09477
- Haarnoja et al. (2018) - SAC - https://arxiv.org/abs/1801.01290

**Tablas HTML:**
- Algorithm Overview (Type, Actions, Policy, Stability)
- Algorithm Selection Guide (Best For, Avoid If, Key Hyperparameter)
- Common Pitfalls & Solutions (Problem → Solution)
- Implementation Checklist (10 debugging steps)

**Recursos Adicionales:**
- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3
- OpenAI Spinning Up: https://spinningup.openai.com/
- Ray RLlib: https://docs.ray.io/en/latest/rllib/

---

## ✅ Algoritmos Implementados

### 🔵 Clásicos (Tabulares)

| Algoritmo | Archivo | Tests | Notebook |
|-----------|---------|-------|----------|
| **Policy Iteration** | `02_algoritmos_clasicos/dynamic_programming/policy_iteration.py` | ✅ | 01 |
| **Value Iteration** | `02_algoritmos_clasicos/dynamic_programming/value_iteration.py` | ✅ | 01 |
| **MC Prediction** | `02_algoritmos_clasicos/monte_carlo/mc_prediction.py` | ✅ | 02 |
| **MC Control** | `02_algoritmos_clasicos/monte_carlo/mc_control.py` | ✅ | 02 |
| **Q-Learning** | `02_algoritmos_clasicos/temporal_difference/q_learning.py` | ✅ | 03 |
| **SARSA** | `02_algoritmos_clasicos/temporal_difference/sarsa.py` | ✅ | 03 |

### 🟢 Deep Reinforcement Learning

| Algoritmo | Archivo | Tests | Notebook | Paper |
|-----------|---------|-------|----------|-------|
| **DQN** | `03_deep_rl/dqn/dqn_basic.py` | ✅ | 04 | Mnih+ 2015 |
| **Double DQN** | `03_deep_rl/dqn/double_dqn.py` | ✅ | 04 | van Hasselt+ 2015 |
| **Dueling DQN** | `03_deep_rl/dqn/dueling_dqn.py` | ✅ | 04 | Wang+ 2016 |
| **REINFORCE** | `03_deep_rl/policy_gradient/reinforce.py` | ✅ | 05 | Williams 1992 |
| **A2C** | `03_deep_rl/policy_gradient/actor_critic.py` | ✅ | 05 | Mnih+ 2016 |

### 🟣 State-of-the-Art (SOTA)

| Algoritmo | Archivo | Tests | Notebook | Paper | Citations |
|-----------|---------|-------|----------|-------|-----------|
| **PPO** | `03_deep_rl/advanced/ppo.py` | ✅ | 06 | Schulman+ 2017 | 3500+ |
| **DDPG** | `03_deep_rl/advanced/ddpg.py` | ✅ | 06 | Lillicrap+ 2016 | Foundation |
| **TD3** | `03_deep_rl/advanced/td3.py` | ✅ | 06 | Fujimoto+ 2018 | Robustness |
| **SAC** | `03_deep_rl/advanced/sac.py` | ✅ | 06 | Haarnoja+ 2018 | SOTA |

---

## 🚀 Inicio Rápido

### 1️⃣ Clonar e Instalar

```bash
# Clonar repositorio
git clone https://github.com/MARK-126/Reinforcement-learning-guide.git
cd Reinforcement-learning-guide

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2️⃣ Verificar Instalación

```bash
# Ejecutar tests
pytest tests/ -v

# Debería mostrar: 100+ tests passed ✅
```

### 3️⃣ Comenzar con los Notebooks

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir: notebooks/01_dynamic_programming_tutorial.ipynb
```

### 4️⃣ Ejecutar un Ejemplo Completo

```bash
# Entrenar DQN en CartPole
cd 04_ejemplos/lunar_lander
python train_dqn.py

# Entrenar PPO en LunarLander
python train_advanced.py
```

---

## 📁 Estructura del Repositorio

```
Reinforcement-learning-guide/
│
├── 📚 notebooks/                      # 6 Notebooks Profesionales (260 KB)
│   ├── 01_dynamic_programming_tutorial.ipynb    # DP (47 KB, 5 ejercicios)
│   ├── 02_monte_carlo_tutorial.ipynb            # MC (31 KB, 5 ejercicios)
│   ├── 03_td_learning_tutorial.ipynb            # TD (51 KB, 4 ejercicios)
│   ├── 04_deep_rl_dqn_tutorial.ipynb            # DQN (44 KB, 5 ejercicios)
│   ├── 05_policy_gradient_tutorial.ipynb        # PG (42 KB, 5 ejercicios)
│   ├── 06_advanced_algorithms_tutorial.ipynb    # SOTA (45 KB, 5 ejercicios)
│   │
│   ├── 🛠️ Utils Modules (87 KB)
│   ├── dp_utils.py              # Dynamic Programming helpers + tests (16 KB)
│   ├── mc_utils.py              # Monte Carlo helpers + tests (15 KB)
│   ├── td_utils.py              # TD Learning agents + tests (16 KB)
│   ├── dqn_utils.py             # DQN networks + tests (16 KB)
│   ├── pg_utils.py              # Policy Gradient tests (9 KB)
│   └── advanced_utils.py        # SOTA algorithms utilities (15 KB)
│
├── 🎯 02_algoritmos_clasicos/         # Implementaciones Clásicas (3,500+ líneas)
│   ├── dynamic_programming/
│   │   ├── policy_iteration.py       # Policy Iteration (396 líneas)
│   │   └── value_iteration.py        # Value Iteration (429 líneas)
│   ├── monte_carlo/
│   │   ├── mc_prediction.py          # First/Every-Visit (612 líneas)
│   │   └── mc_control.py             # On/Off-Policy (852 líneas)
│   └── temporal_difference/
│       ├── q_learning.py             # Q-Learning off-policy
│       ├── sarsa.py                  # SARSA on-policy
│       └── td_lambda.py              # TD(λ) eligibility traces
│
├── 🧠 03_deep_rl/                     # Deep RL (16,000+ líneas)
│   ├── dqn/
│   │   ├── dqn_basic.py              # DQN básico
│   │   ├── double_dqn.py             # Double DQN (595 líneas)
│   │   ├── dueling_dqn.py            # Dueling DQN (744 líneas)
│   │   ├── README.md                 # Documentación DQN
│   │   └── EXAMPLES.md               # Ejemplos de uso
│   ├── policy_gradient/
│   │   ├── reinforce.py              # REINFORCE (720 líneas)
│   │   ├── actor_critic.py           # A2C + GAE (916 líneas)
│   │   └── README.md
│   └── advanced/                     # 🌟 SOTA Algorithms
│       ├── ppo.py                    # PPO (963 líneas)
│       ├── ddpg.py                   # DDPG
│       ├── td3.py                    # TD3
│       ├── sac.py                    # SAC (928 líneas)
│       ├── README.md                 # Comparación SOTA
│       └── EXAMPLES.md
│
├── 🎮 04_ejemplos/                    # Proyectos Completos
│   ├── cartpole/
│   │   └── cartpole_qlearning.py
│   ├── lunar_lander/
│   │   ├── train_dqn.py              # DQN training script (333 líneas)
│   │   ├── train_advanced.py         # PPO training (319 líneas)
│   │   └── README.md
│   └── mountain_car/
│
├── 🧪 tests/                          # Suite de Tests (100+ tests)
│   ├── test_algorithms_clasicos/
│   │   ├── test_dynamic_programming.py  # 20+ tests
│   │   └── test_monte_carlo.py          # 30+ tests
│   └── pytest.ini
│
├── 🔧 utils/                          # Utilidades Generales
│   ├── plotting.py
│   └── replay_buffer.py
│
├── ⚙️ .github/workflows/
│   └── tests.yml                     # CI/CD GitHub Actions
│
├── 📄 Documentation
│   ├── README.md                     # Este archivo
│   ├── RESTRUCTURE_SUMMARY.md        # Resumen de reestructuración
│   ├── CONTRIBUTING.md               # Guía de contribución
│   ├── QUICKSTART.md                 # Inicio rápido (30 min)
│   └── requirements.txt              # Dependencias
│
└── 📋 Configuration
    ├── pytest.ini                    # Configuración de tests
    ├── .gitignore
    └── LICENSE                       # MIT License
```

---

## 🎓 Ruta de Aprendizaje

### 📅 Nivel 1: Fundamentos (3-4 semanas)

**Semana 1-2: Conceptos Básicos**
- [ ] Leer `01_fundamentos/` (MDPs, Bellman, Value Functions)
- [ ] Completar **Notebook 01** (Dynamic Programming)
  - [ ] Ejercicio 1: Policy Evaluation
  - [ ] Ejercicio 2: Policy Improvement
  - [ ] Ejercicio 3: Value Iteration Step
  - [ ] Ejercicio 4: Extract Policy
  - [ ] Ejercicio 5: Compare Algorithms
- [ ] Ejecutar todos los tests de DP

**Semana 3-4: Métodos Model-Free**
- [ ] Completar **Notebook 02** (Monte Carlo)
  - [ ] Ejercicios 1-2: First/Every-Visit MC
  - [ ] Ejercicios 3-4: On/Off-Policy Control
  - [ ] Ejercicio 5: Importance Sampling
- [ ] Completar **Notebook 03** (TD Learning)
  - [ ] Ejercicios 1-3: Q-Learning, SARSA, Expected SARSA
  - [ ] Ejercicio 4: Compare TD Methods

**Proyecto Práctico:** Resolver GridWorld con Q-Learning

---

### 📅 Nivel 2: Deep Reinforcement Learning (4-6 semanas)

**Semana 5-7: Deep Q-Networks**
- [ ] Repasar PyTorch básico (si es necesario)
- [ ] Completar **Notebook 04** (DQN)
  - [ ] Ejercicio 1: DQN Network
  - [ ] Ejercicio 2: Replay Buffer
  - [ ] Ejercicio 3: DQN Update
  - [ ] Ejercicio 4: Double DQN
  - [ ] Ejercicio 5: Dueling DQN
- [ ] Leer papers: Mnih+ 2015, van Hasselt+ 2015, Wang+ 2016
- [ ] Entrenar agente en CartPole

**Semana 8-10: Policy Gradients**
- [ ] Completar **Notebook 05** (Policy Gradient)
  - [ ] Ejercicio 1: Policy Network
  - [ ] Ejercicio 2: REINFORCE Loss
  - [ ] Ejercicio 3: Baseline
  - [ ] Ejercicio 4: Actor-Critic
  - [ ] Ejercicio 5: GAE
- [ ] Leer papers: Williams 1992, Mnih+ 2016 (A3C)
- [ ] Entrenar A2C en Pendulum

**Proyecto Práctico:** Resolver LunarLander con DQN/Double DQN

---

### 📅 Nivel 3: State-of-the-Art (6-8 semanas)

**Semana 11-14: Algoritmos SOTA**
- [ ] Completar **Notebook 06** (Advanced)
  - [ ] Ejercicio 1: PPO Loss
  - [ ] Ejercicio 2: DDPG Networks
  - [ ] Ejercicio 3: TD3 Updates
  - [ ] Ejercicio 4: SAC Temperature
  - [ ] Ejercicio 5: Compare SOTA
- [ ] Leer los 4 papers SOTA (Schulman, Lillicrap, Fujimoto, Haarnoja)
- [ ] Implementar cada algoritmo desde cero
- [ ] Comparar performance en múltiples ambientes

**Semana 15-18: Proyectos Avanzados**
- [ ] Implementar algoritmo desde paper reciente
- [ ] Crear ambiente custom
- [ ] Aplicar RL a problema real
- [ ] Contribuir a proyecto open-source

**Proyecto Final:** Sistema completo de RL con PPO/SAC en ambiente complejo

---

### 🎯 Objetivos por Nivel

| Nivel | Algoritmos Dominados | Skills Adquiridos | Tiempo Estimado |
|-------|---------------------|-------------------|-----------------|
| **1. Fundamentos** | DP, MC, Q-Learning, SARSA | MDPs, Bellman, value functions | 3-4 semanas |
| **2. Deep RL** | DQN, Double DQN, Dueling, A2C | PyTorch, neural nets, experience replay | 4-6 semanas |
| **3. SOTA** | PPO, DDPG, TD3, SAC | Arquitecturas avanzadas, tuning | 6-8 semanas |
| **Total** | 15+ algoritmos | RL profesional completo | 13-18 semanas |

---

## 🔬 Referencias Académicas

### 📄 Papers Fundamentales Citados

#### Dynamic Programming
- **Bellman, R.** (1957). *Dynamic Programming*. Princeton University Press.
- **Puterman, M.** (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
- **Sutton & Barto** (2018). *Reinforcement Learning: An Introduction* (2nd Edition). MIT Press. [Free Online](http://incompleteideas.net/book/the-book-2nd.html)

#### Deep Q-Learning
1. **Mnih, V., et al.** (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
   - **Link:** [Nature Paper](https://www.nature.com/articles/nature14236)
   - **Citations:** 10,000+
   - **Key Innovation:** Experience Replay + Target Networks

2. **van Hasselt, H., Guez, A., & Silver, D.** (2015). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.
   - **Link:** https://arxiv.org/abs/1509.06461
   - **Key Innovation:** Reduces Q-value overestimation

3. **Wang, Z., et al.** (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *ICML*.
   - **Link:** https://arxiv.org/abs/1511.06581
   - **Key Innovation:** Separate Value and Advantage streams

#### Policy Gradient Methods
4. **Williams, R. J.** (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3), 229-256.
   - **Algorithm:** REINFORCE

5. **Mnih, V., et al.** (2016). "Asynchronous methods for deep reinforcement learning." *ICML*.
   - **Link:** https://arxiv.org/abs/1602.01783
   - **Algorithm:** A3C (Asynchronous Advantage Actor-Critic)

#### State-of-the-Art Algorithms
6. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). "Proximal Policy Optimization Algorithms."
   - **Link:** https://arxiv.org/abs/1707.06347
   - **Citations:** 3,500+
   - **Algorithm:** PPO (most popular RL algorithm)

7. **Lillicrap, T. P., et al.** (2016). "Continuous control with deep reinforcement learning." *ICLR*.
   - **Link:** https://arxiv.org/abs/1509.02971
   - **Algorithm:** DDPG (foundation for continuous control)

8. **Fujimoto, S., van Hoof, H., & Meger, D.** (2018). "Addressing Function Approximation Error in Actor-Critic Methods." *ICML*.
   - **Link:** https://arxiv.org/abs/1802.09477
   - **Algorithm:** TD3 (improves DDPG stability)

9. **Haarnoja, T., et al.** (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor."
   - **Link:** https://arxiv.org/abs/1801.01290
   - **Algorithm:** SAC (current SOTA for continuous control)

---

## 📖 Recursos Adicionales

### 📚 Libros Recomendados

1. **"Reinforcement Learning: An Introduction"** - Sutton & Barto (2018)
   - El libro de texto definitivo
   - [Versión gratuita](http://incompleteideas.net/book/the-book-2nd.html)
   - Cubre todos los fundamentos y algoritmos clásicos

2. **"Deep Reinforcement Learning Hands-On"** - Maxim Lapan
   - Enfoque práctico con PyTorch
   - 2nd Edition actualizada (2020)

3. **"Algorithms for Reinforcement Learning"** - Csaba Szepesvári
   - Perspectiva matemática rigurosa
   - [PDF gratuito](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)

4. **"Grokking Deep Reinforcement Learning"** - Miguel Morales
   - Explicaciones intuitivas con visualizaciones

### 🎥 Cursos Online

1. **David Silver's RL Course** (DeepMind)
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
   - 10 lectures covering fundamentals to AlphaGo
   - Nivel: Intermedio

2. **CS285: Deep Reinforcement Learning** - UC Berkeley (Sergey Levine)
   - [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)
   - Graduate-level course
   - Actualizado anualmente

3. **Spinning Up in Deep RL** - OpenAI
   - [Interactive Docs](https://spinningup.openai.com/)
   - Excellent introduction to deep RL
   - Includes code implementations

4. **Deep RL Bootcamp** (2017)
   - [Video Lectures](https://sites.google.com/view/deep-rl-bootcamp/lectures)
   - 2-day intensive bootcamp by top researchers

### 🛠️ Librerías y Frameworks

- **Stable Baselines3** - https://github.com/DLR-RM/stable-baselines3
  - Implementaciones profesionales de RL algorithms
  - Mejor para producción

- **Ray RLlib** - https://docs.ray.io/en/latest/rllib/
  - Distributed RL at scale
  - Integración con Ray

- **Gymnasium** - https://gymnasium.farama.org/
  - Fork mantenido de OpenAI Gym
  - Environments estándar

- **CleanRL** - https://github.com/vwxyzjn/cleanrl
  - Single-file implementations
  - Excellent for learning

### 🌐 Comunidades

- **r/reinforcementlearning** - https://reddit.com/r/reinforcementlearning
  - Subreddit activo con 50k+ miembros

- **RL Discord** - https://discord.gg/xhfNqQv
  - Comunidad en tiempo real

- **Papers with Code** - https://paperswithcode.com/area/reinforcement-learning
  - Papers + implementaciones

---

## 🤝 Contribuciones

¡Las contribuciones son muy bienvenidas! Este es un proyecto educativo open-source.

### Cómo Contribuir

1. **Fork** el repositorio
2. Crea una **rama** para tu feature (`git checkout -b feature/mejora-notebook`)
3. **Commit** tus cambios (`git commit -m 'feat: Añadir ejercicio de Multi-Armed Bandits'`)
4. **Push** a la rama (`git push origin feature/mejora-notebook`)
5. Abre un **Pull Request**

### Áreas de Contribución

- 📝 Mejorar explicaciones en notebooks
- 🧪 Añadir más tests
- 🎯 Crear nuevos ejemplos/proyectos
- 📚 Traducir a otros idiomas
- 🐛 Reportar y corregir bugs
- 📊 Añadir visualizaciones
- 🔬 Implementar nuevos algoritmos

### Guidelines

- Seguir el estilo de código existente (PEP 8)
- Añadir tests para nuevo código
- Actualizar documentación
- Commits descriptivos en español o inglés

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para más detalles.

---

## 🙏 Agradecimientos

Este repositorio fue posible gracias a:

- **Sutton & Barto** por el libro fundamental de RL
- **OpenAI** por Gym/Gymnasium y Spinning Up
- **DeepMind** por papers foundational (DQN, AlphaGo)
- **UC Berkeley** por el curso CS285
- **DeepLearning.AI** por el formato de notebooks profesionales
- Toda la **comunidad open-source** de RL

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2024 MARK-126

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📞 Contacto

**Autor:** MARK-126

**Repositorio:** https://github.com/MARK-126/Reinforcement-learning-guide

---

## ⭐ Star History

Si este repositorio te resulta útil, **¡considera darle una estrella!** ⭐

Ayuda a otros estudiantes a descubrir este recurso.

---

## 🚀 Próximos Pasos

Después de completar esta guía, estarás preparado para:

✅ **Meta-Learning** - Algoritmos que aprenden a aprender
✅ **Multi-Agent RL** - Sistemas con múltiples agentes
✅ **Model-Based RL** - Aprender modelos del mundo
✅ **Offline RL** - Aprender de datasets fijos
✅ **RL Research** - Contribuir a papers y avances

---

<div align="center">

### ¡Feliz Aprendizaje de Reinforcement Learning! 🤖🎓🚀

**"The only way to do great work is to learn deeply."**

[⬆️ Volver arriba](#-reinforcement-learning-guía-profesional-completa)

</div>
