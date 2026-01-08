# 01_introduccion_practica# 📚 Introducción Práctica

Esta carpeta contiene notebooks prácticos que te introducen a Reinforcement Learning de forma gradual.

---

## 📓 Notebooks Disponibles

### 1. `gym_basics.ipynb` ⏱️ 45 minutos

**Objetivo**: Aprender a usar Gymnasium (antes OpenAI Gym).

**Lo que aprenderás**:
- ✅ Crear ambientes
- ✅ Interactuar con el ambiente (reset, step)
- ✅ Entender observaciones, acciones y recompensas

**Ejercicios**: 5 ejercicios simples (1-8 líneas)

**Nivel**: Principiante 🟢

---

### 2. `simple_q_learning.ipynb` ⏱️ 1 hora

**Objetivo**: Implementar Q-Learning desde cero en FrozenLake.

**Lo que aprenderás**:
- ✅ Crear Q-tables
- ✅ Política epsilon-greedy
- ✅ Actualizar Q-values
- ✅ Entrenar un agente completo

**Ejercicios**: 4 ejercicios (1-15 líneas)

**Nivel**: Principiante 🟢

**Prerequisito**: `gym_basics.ipynb`

---

### 3. `simple_dqn.ipynb` ⏱️ 1 hora ⭐ NUEVO

**Objetivo**: Tu primer Deep Q-Network - puente a Deep RL.

**Lo que aprenderás**:
- ✅ Crear red Q con PyTorch
- ✅ Aproximar Q-values con redes neuronales
- ✅ Entrenar DQN básico en CartPole
- ✅ Entender diferencia entre Q-table y Q-network

**Ejercicios**: 5 ejercicios (2-12 líneas)

**Nivel**: Intermedio 🟡

**Prerequisitos**: 
- `pytorch_for_rl.ipynb` (en 00_prerequisites)
- `simple_q_learning.ipynb`

---

## 🎯 Orden Recomendado

1. **`gym_basics.ipynb`** ← Empieza aquí
2. **`simple_q_learning.ipynb`** ← Q-Learning tabular
3. **`simple_dqn.ipynb`** ← Puente a Deep RL

**Tiempo total**: ~2.75 horas

---

## ✅ Después de Completar

Continúa con:
- `02_algoritmos_clasicos/` - Para algoritmos clásicos más avanzados
- `03_deep_rl/` - Para Deep RL avanzado (DQN completo, Double DQN, etc.)

---

## 📋 Requisitos Previos

Antes de empezar estos notebooks, completa:
- `00_prerequisites/numpy_for_rl.ipynb`

---

## 📦 Instalación

```bash
pip install gymnasium numpy
```

---

## 🚀 Próximo Paso

Después de completar estos notebooks, continúa con:
👉 `02_algoritmos_clasicos/temporal_difference/Q_Learning_Exercise.ipynb`
