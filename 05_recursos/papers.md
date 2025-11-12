# Papers Fundamentales en Reinforcement Learning

## Papers Clásicos

### 1. Temporal Difference Learning
**"Learning to Predict by the Methods of Temporal Differences"**
- Autor: Richard S. Sutton (1988)
- [Link al paper](https://link.springer.com/article/10.1007/BF00115009)
- **Importancia**: Introduce TD Learning, base de muchos algoritmos modernos
- **Conceptos clave**: TD error, bootstrapping

### 2. Q-Learning
**"Learning from Delayed Rewards"**
- Autor: Christopher J.C.H. Watkins (1989) - PhD Thesis
- [Link](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)
- **Importancia**: Introduce Q-Learning, primer algoritmo off-policy comprobadamente convergente
- **Conceptos clave**: Q-values, off-policy learning

### 3. Policy Gradient Theorem
**"Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"**
- Autor: Ronald J. Williams (1992)
- [Link](https://link.springer.com/article/10.1007/BF00992696)
- **Importancia**: Fundamenta métodos de gradiente de política
- **Conceptos clave**: REINFORCE, policy gradient

## Deep Reinforcement Learning Era

### 4. Deep Q-Network (DQN)
**"Playing Atari with Deep Reinforcement Learning"**
- Autores: Mnih et al. (DeepMind, 2013)
- [Link al paper (arXiv)](https://arxiv.org/abs/1312.5602)

**"Human-level control through deep reinforcement learning"**
- Autores: Mnih et al. (Nature, 2015)
- [Link al paper](https://www.nature.com/articles/nature14236)
- **Importancia**: Primera demostración de DRL jugando Atari a nivel humano
- **Conceptos clave**: Experience replay, target networks, CNN para estados visuales

### 5. Double DQN
**"Deep Reinforcement Learning with Double Q-learning"**
- Autores: van Hasselt, Guez, Silver (2015)
- [Link](https://arxiv.org/abs/1509.06461)
- **Importancia**: Reduce sobreestimación de valores en DQN
- **Conceptos clave**: Decoupling selection and evaluation

### 6. Dueling DQN
**"Dueling Network Architectures for Deep Reinforcement Learning"**
- Autores: Wang et al. (2016)
- [Link](https://arxiv.org/abs/1511.06581)
- **Importancia**: Separa estimación de V(s) y A(s,a)
- **Conceptos clave**: Advantage function, value stream, advantage stream

### 7. Prioritized Experience Replay
**"Prioritized Experience Replay"**
- Autores: Schaul et al. (2016)
- [Link](https://arxiv.org/abs/1511.05952)
- **Importancia**: Mejora sample efficiency priorizando transiciones importantes
- **Conceptos clave**: TD error como prioridad, importance sampling

## Policy Gradient Methods

### 8. Trust Region Policy Optimization (TRPO)
**"Trust Region Policy Optimization"**
- Autores: Schulman et al. (2015)
- [Link](https://arxiv.org/abs/1502.05477)
- **Importancia**: Garantiza mejoras monotónicas de política
- **Conceptos clave**: Trust region, KL constraint, natural gradient

### 9. Proximal Policy Optimization (PPO)
**"Proximal Policy Optimization Algorithms"**
- Autores: Schulman et al. (OpenAI, 2017)
- [Link](https://arxiv.org/abs/1707.06347)
- **Importancia**: Simplifica TRPO, se convierte en estándar de la industria
- **Conceptos clave**: Clipped objective, multiple epochs per batch
- **Por qué es importante**: Balance perfecto entre performance y simplicidad

### 10. Asynchronous Advantage Actor-Critic (A3C)
**"Asynchronous Methods for Deep Reinforcement Learning"**
- Autores: Mnih et al. (2016)
- [Link](https://arxiv.org/abs/1602.01783)
- **Importancia**: Paralelización eficiente sin replay buffer
- **Conceptos clave**: Asynchronous updates, n-step returns, entropy regularization

## Actor-Critic Methods

### 11. Deterministic Policy Gradient (DPG)
**"Deterministic Policy Gradient Algorithms"**
- Autores: Silver et al. (2014)
- [Link](http://proceedings.mlr.press/v32/silver14.pdf)
- **Importancia**: Extiende policy gradient a acciones continuas
- **Conceptos clave**: Deterministic policies, compatible function approximation

### 12. Deep Deterministic Policy Gradient (DDPG)
**"Continuous control with deep reinforcement learning"**
- Autores: Lillicrap et al. (2016)
- [Link](https://arxiv.org/abs/1509.02971)
- **Importancia**: DPG con deep learning para control continuo
- **Conceptos clave**: Actor-critic, off-policy, replay buffer

### 13. Twin Delayed DDPG (TD3)
**"Addressing Function Approximation Error in Actor-Critic Methods"**
- Autores: Fujimoto et al. (2018)
- [Link](https://arxiv.org/abs/1802.09477)
- **Importancia**: Mejora estabilidad de DDPG
- **Conceptos clave**: Twin critics, delayed policy updates, target policy smoothing

### 14. Soft Actor-Critic (SAC)
**"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"**
- Autores: Haarnoja et al. (2018)
- [Link](https://arxiv.org/abs/1801.01290)
- **Importancia**: Estado del arte en control continuo
- **Conceptos clave**: Maximum entropy RL, automatic temperature tuning

## Model-Based RL

### 15. World Models
**"World Models"**
- Autores: Ha, Schmidhuber (2018)
- [Link](https://arxiv.org/abs/1803.10122)
- **Importancia**: Aprende modelo del mundo con VAE
- **Conceptos clave**: Vision model, memory model, controller

### 16. MuZero
**"Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"**
- Autores: Schrittwieser et al. (DeepMind, 2020)
- [Link](https://arxiv.org/abs/1911.08265)
- **Importancia**: SOTA en juegos sin conocer las reglas
- **Conceptos clave**: Learned dynamics, planning, self-play

## Multi-Agent RL

### 17. Multi-Agent DDPG
**"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"**
- Autores: Lowe et al. (2017)
- [Link](https://arxiv.org/abs/1706.02275)
- **Importancia**: Extiende DDPG a entornos multi-agente
- **Conceptos clave**: Centralized training, decentralized execution

## Landmark Applications

### 18. AlphaGo
**"Mastering the game of Go with deep neural networks and tree search"**
- Autores: Silver et al. (Nature, 2016)
- [Link](https://www.nature.com/articles/nature16961)
- **Importancia**: Primer programa en vencer a campeón mundial de Go
- **Conceptos clave**: MCTS, policy network, value network, self-play

### 19. AlphaZero
**"A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"**
- Autores: Silver et al. (Science, 2018)
- [Link](https://science.sciencemag.org/content/362/6419/1140)
- **Importancia**: Generaliza AlphaGo a múltiples juegos
- **Conceptos clave**: Tabula rasa learning, self-play only

### 20. OpenAI Five
**"Dota 2 with Large Scale Deep Reinforcement Learning"**
- Autores: Berner et al. (OpenAI, 2019)
- [Link](https://arxiv.org/abs/1912.06680)
- **Importancia**: Derrota a jugadores profesionales en Dota 2
- **Conceptos clave**: PPO at scale, curriculum learning, surgery

## Exploration

### 21. Intrinsic Curiosity Module
**"Curiosity-driven Exploration by Self-supervised Prediction"**
- Autores: Pathak et al. (2017)
- [Link](https://arxiv.org/abs/1705.05363)
- **Importancia**: Recompensa intrínseca basada en predicción
- **Conceptos clave**: Forward model, inverse model

### 22. Random Network Distillation
**"Exploration by Random Network Distillation"**
- Autores: Burda et al. (OpenAI, 2018)
- [Link](https://arxiv.org/abs/1810.12894)
- **Importancia**: Exploración simple pero efectiva
- **Conceptos clave**: Novelty detection, prediction error

## Offline RL

### 23. Conservative Q-Learning (CQL)
**"Conservative Q-Learning for Offline Reinforcement Learning"**
- Autores: Kumar et al. (2020)
- [Link](https://arxiv.org/abs/2006.04779)
- **Importancia**: Aprende de datasets fijos sin interacción
- **Conceptos clave**: Conservative value estimation, regularization

## Meta-RL

### 24. MAML
**"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"**
- Autores: Finn et al. (2017)
- [Link](https://arxiv.org/abs/1703.03400)
- **Importancia**: Meta-learning aplicable a RL
- **Conceptos clave**: Few-shot adaptation, meta-optimization

## Surveys y Reviews

### 25. Deep RL Review
**"Deep Reinforcement Learning: An Overview"**
- Autores: Li (2017)
- [Link](https://arxiv.org/abs/1701.07274)
- Excelente overview de métodos de Deep RL

### 26. RL Tutorial
**"Reinforcement Learning: A Survey"**
- Autores: Kaelbling, Littman, Moore (1996)
- [Link](https://www.jair.org/index.php/jair/article/view/10166)
- Survey clásico de RL tradicional

## Cómo Leer Papers de RL

### Estrategia de Lectura

1. **Primera pasada** (10-15 min):
   - Abstract y conclusión
   - Figuras y resultados principales
   - ¿Vale la pena leer en detalle?

2. **Segunda pasada** (30-60 min):
   - Introducción completa
   - Método (sin todos los detalles matemáticos)
   - Experimentos y resultados
   - Tomar notas

3. **Tercera pasada** (2-3 horas):
   - Entender cada ecuación
   - Pseudocódigo del algoritmo
   - Implementar si es relevante

### Qué Buscar

- **Motivación**: ¿Qué problema resuelve?
- **Contribución**: ¿Qué es nuevo?
- **Método**: ¿Cómo funciona?
- **Resultados**: ¿Funciona realmente?
- **Limitaciones**: ¿Qué no puede hacer?

## Papers por Orden de Lectura Recomendado

### Nivel Principiante
1. DQN (2015)
2. Policy Gradient Theorem (1992)
3. A3C (2016)

### Nivel Intermedio
4. PPO (2017)
5. DDPG (2016)
6. Double DQN (2015)
7. Dueling DQN (2016)

### Nivel Avanzado
8. TRPO (2015)
9. SAC (2018)
10. TD3 (2018)
11. MuZero (2020)

## Recursos para Encontrar Papers

- **arXiv.org**: Preprints de papers más recientes
- **Google Scholar**: Búsqueda académica
- **Papers With Code**: Papers con implementaciones
- **Semantic Scholar**: Búsqueda con AI
- **Twitter**: Autores comparten sus papers

## Implementaciones de Referencia

- **Stable-Baselines3**: Implementaciones de alta calidad en PyTorch
- **OpenAI Baselines**: Implementaciones de referencia (TensorFlow)
- **CleanRL**: Implementaciones simples y limpias
- **RLlib**: Framework escalable de Ray

---

**Consejo**: No intentes leer todos los papers. Enfócate en los fundamentales para tu área de interés y lee en profundidad.
