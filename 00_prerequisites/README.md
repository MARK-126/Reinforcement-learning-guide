# 📚 Prerequisites para Reinforcement Learning

## 🎯 ¿Para Quién es Esta Sección?

Esta sección está diseñada para personas con **CERO conocimientos previos** en:
- Matemáticas (más allá de aritmética básica)
- Programación
- Machine Learning
- Reinforcement Learning

**Si ya tienes experiencia**, puedes saltar directamente a [Fundamentos de RL](../01_fundamentos/).

---

## 🗺️ Mapa de Contenido

```
00_prerequisites/
│
├── 01_matematicas_basicas.md      ← Empieza aquí
│   ├── Probabilidad y estadística
│   ├── Notación matemática
│   ├── Valor esperado
│   ├── Distribuciones
│   └── Descuento geométrico
│
├── 02_algebra_lineal.md
│   ├── Vectores y operaciones
│   ├── Matrices y multiplicación
│   ├── Producto punto y normas
│   ├── Ecuación de Bellman matricial
│   └── NumPy para álgebra lineal
│
├── 03_calculo_basico.md
│   ├── Derivadas desde cero
│   ├── Funciones de activación
│   ├── Derivadas parciales
│   ├── Gradientes
│   ├── Chain rule
│   ├── Gradient descent
│   └── Backpropagation
│
├── 04_python_numpy.md
│   ├── Python esencial
│   ├── Estructuras de datos
│   ├── Control de flujo
│   ├── Funciones y clases
│   ├── NumPy completo
│   ├── Matplotlib básico
│   └── Código práctico para RL
│
└── 05_conceptos_optimizacion.md
    ├── Gradient descent
    ├── SGD, momentum, Adam
    ├── Learning rate schedules
    ├── Gradient clipping
    ├── Optimización en Deep RL
    └── Hiperparámetros
```

---

## 📖 Orden de Estudio Recomendado

### 🟢 Ruta Completa (Principiante Absoluto)

**Semana 1-2**: Matemáticas Básicas
- [ ] [01_matematicas_basicas.md](01_matematicas_basicas.md)
- [ ] Completar todos los ejercicios
- [ ] Implementar funciones en Python
- **Tiempo estimado**: 10-15 horas

**Semana 2-3**: Álgebra Lineal
- [ ] [02_algebra_lineal.md](02_algebra_lineal.md)
- [ ] Practicar operaciones en NumPy
- [ ] Resolver ecuación de Bellman matricial
- **Tiempo estimado**: 10-15 horas

**Semana 3-4**: Cálculo
- [ ] [03_calculo_basico.md](03_calculo_basico.md)
- [ ] Entender derivadas y gradientes
- [ ] Implementar gradient descent
- **Tiempo estimado**: 10-15 horas

**Semana 4-5**: Python y NumPy
- [ ] [04_python_numpy.md](04_python_numpy.md)
- [ ] Escribir código práctico
- [ ] Implementar estructuras de datos para RL
- **Tiempo estimado**: 15-20 horas

**Semana 5-6**: Optimización
- [ ] [05_conceptos_optimizacion.md](05_conceptos_optimizacion.md)
- [ ] Comparar optimizadores
- [ ] Experimentar con learning rates
- **Tiempo estimado**: 8-12 horas

**Total**: ~6 semanas (55-77 horas)

### 🟡 Ruta Acelerada (Con Algo de Experiencia)

Si ya sabes programación básica:
1. Revisar rápido: 01, 02, 03 (3-5 días)
2. Enfocarse en: 04, 05 (1 semana)
3. Pasar a [Fundamentos de RL](../01_fundamentos/)

**Total**: ~2 semanas

### 🔴 Solo Refresco (Experiencia en ML)

Si ya conoces ML:
1. Hojear cada documento para recordar notación
2. Enfocarse en diferencias específicas de RL
3. Ir directo a tutoriales de RL

**Total**: 2-3 días

---

## 🎓 Objetivos de Aprendizaje

Al completar esta sección, podrás:

### Matemáticas
✅ Calcular probabilidades y esperanzas
✅ Trabajar con distribuciones (normal, uniforme, Bernoulli)
✅ Entender notación matemática (Σ, argmax, E[·])
✅ Calcular retornos descontados

### Álgebra Lineal
✅ Operar con vectores y matrices
✅ Calcular productos punto y matriciales
✅ Usar NumPy para álgebra lineal
✅ Entender ecuación de Bellman en forma matricial

### Cálculo
✅ Calcular derivadas básicas
✅ Aplicar chain rule
✅ Computar gradientes
✅ Implementar gradient descent
✅ Entender backpropagation conceptualmente

### Programación
✅ Escribir Python funcional
✅ Usar NumPy eficientemente
✅ Implementar clases para agentes
✅ Visualizar resultados con Matplotlib
✅ Trabajar con Gymnasium (OpenAI Gym)

### Optimización
✅ Entender diferentes optimizadores (SGD, Adam)
✅ Usar learning rate schedules
✅ Aplicar gradient clipping
✅ Debuggear problemas de optimización

---

## 💡 Consejos para el Estudio

### 1. Practica Activamente

❌ **Mal**: Solo leer pasivamente
✅ **Bien**: Implementar cada concepto en código

**Ejemplo**:
```python
# Después de leer sobre valor esperado, impleméntalo
def valor_esperado(valores, probabilidades):
    return sum(v * p for v, p in zip(valores, probabilidades))

# Test
valores = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6
print(f"E[dado] = {valor_esperado(valores, probs)}")  # 3.5
```

### 2. Haz Todos los Ejercicios

Cada documento tiene ejercicios prácticos. **Hazlos todos.**

### 3. Usa Jupyter Notebooks

```bash
# Instala Jupyter
pip install jupyter

# Crea notebook para cada tema
jupyter notebook matematicas_basicas_practica.ipynb
```

### 4. Consulta Recursos Externos

Cada documento tiene sección de recursos adicionales. Úsala.

### 5. No Te Atasques

Si algo no tiene sentido después de 30 minutos:
1. Toma un descanso
2. Busca explicación alternativa (YouTube, Khan Academy)
3. Sigue adelante y regresa después

### 6. Forma un Grupo de Estudio

Explica conceptos a otros. Si puedes enseñarlo, lo entendiste.

---

## 🛠️ Setup del Entorno

### Instalación Básica

```bash
# Python 3.8+
python --version

# Crear entorno virtual
python -m venv rl_env
source rl_env/bin/activate  # Linux/Mac
# o
rl_env\Scripts\activate  # Windows

# Instalar dependencias
pip install numpy matplotlib jupyter
pip install gymnasium torch
```

### Verificar Instalación

```python
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("Gymnasium:", gym.__version__)

# Test
env = gym.make('CartPole-v1')
state, info = env.reset()
print("Estado inicial:", state)
```

---

## 📊 Autoevaluación

Antes de pasar a RL, verifica que puedes:

### Test Rápido de Matemáticas

```python
# 1. Calcular P(A y B) si P(A)=0.3, P(B|A)=0.7
# Respuesta: 0.21

# 2. ¿Qué es argmax([0.2, 0.8, 0.5, 0.9])?
# Respuesta: 3

# 3. Calcular Σᵢ₌₁⁵ i²
# Respuesta: 55

# 4. Si recompensas = [1, 1, 1], γ=0.9, ¿cuál es G₀?
# Respuesta: 1 + 0.9 + 0.81 = 2.71
```

### Test Rápido de Álgebra Lineal

```python
import numpy as np

# 1. Producto punto de [1, 2, 3] y [4, 5, 6]
# Respuesta: 32

# 2. Shape de matriz A(3x4) @ vector v(4x1)
# Respuesta: (3x1)

# 3. ¿Qué hace np.argmax([[1, 2], [3, 4]])?
# Respuesta: 3 (índice aplanado del máximo)
```

### Test Rápido de Cálculo

```python
# 1. Derivada de f(x) = x³ + 2x
# Respuesta: f'(x) = 3x² + 2

# 2. Si f(x,y) = x²y, ¿cuál es ∂f/∂x?
# Respuesta: 2xy

# 3. ¿En qué dirección apunta el gradiente?
# Respuesta: Dirección de mayor crecimiento
```

### Test Rápido de Python

```python
# 1. Crear array NumPy de ceros 3x4
# Respuesta: np.zeros((3, 4))

# 2. Obtener elemento máximo de lista
# Respuesta: max(lista) o np.max(array)

# 3. Iterar con índice sobre lista
# Respuesta: for i, item in enumerate(lista):
```

**Si respondiste correctamente 80%+**: ¡Listo para RL!
**Si no**: Revisa las secciones relevantes.

---

## 🔗 Próximos Pasos

Una vez completados los prerequisites:

1. **[Fundamentos de RL](../01_fundamentos/introduccion.md)**
   - ¿Qué es RL?
   - MDPs
   - Ecuaciones de Bellman
   - Value functions y políticas

2. **[Tutorial 01: Dynamic Programming](../notebooks/01_dynamic_programming_tutorial.ipynb)**
   - Policy Evaluation
   - Policy Iteration
   - Value Iteration
   - Implementación práctica

3. **[Tutorial 02: Monte Carlo](../notebooks/02_monte_carlo_tutorial.ipynb)**
   - Métodos model-free
   - On-policy vs off-policy
   - Importance sampling

---

## 📚 Recursos Adicionales

### Cursos Online (Gratis)

**Matemáticas**:
- [Khan Academy - Probabilidad](https://es.khanacademy.org/math/probability)
- [Khan Academy - Estadística](https://es.khanacademy.org/math/statistics-probability)
- [Khan Academy - Álgebra Lineal](https://es.khanacademy.org/math/linear-algebra)
- [Khan Academy - Cálculo](https://es.khanacademy.org/math/calculus-1)

**Python**:
- [Python.org Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)
- [Automate the Boring Stuff](https://automatetheboringstuff.com/)

**NumPy**:
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Tutorial por CS231n](http://cs231n.github.io/python-numpy-tutorial/)

**Machine Learning Math**:
- [Mathematics for Machine Learning (Book)](https://mml-book.github.io/)
- [Deep Learning Book - Math Chapters](https://www.deeplearningbook.org/)

### Videos (YouTube)

- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown - Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [StatQuest - Statistics](https://www.youtube.com/c/joshstarmer)

### Libros

**Básicos**:
- Strang, Gilbert. "Introduction to Linear Algebra"
- Stewart, James. "Calculus"
- Ross, Sheldon. "A First Course in Probability"

**ML Math**:
- Deisenroth et al. "Mathematics for Machine Learning"
- Boyd & Vandenberghe. "Convex Optimization"

---

## ❓ FAQ

### P: ¿Realmente necesito todo esto para RL?

**R**: Depende de tu objetivo:
- **Implementar algoritmos básicos (tabular RL)**: Matemáticas básicas + Python (40% del contenido)
- **Entender papers de RL**: Matemáticas + Álgebra + Cálculo (70%)
- **Implementar Deep RL**: Todo el contenido (100%)

### P: ¿Cuánto tiempo me tomará?

**R**:
- Principiante absoluto: 4-6 semanas (10 hrs/semana)
- Con algo de experiencia: 2-3 semanas
- Solo refresco: 3-5 días

### P: ¿Puedo aprender RL sin cálculo?

**R**: Sí, para **RL tabular** (Q-Learning, SARSA, etc.). Pero **Deep RL** requiere cálculo para entender backpropagation.

### P: ¿Python es obligatorio?

**R**: No estrictamente, pero es el estándar de facto en RL. 95% de implementaciones y papers usan Python.

### P: ¿Qué si me salto los prerequisites?

**R**: Puedes intentarlo, pero:
- No entenderás la matemática detrás de los algoritmos
- Tendrás problemas implementando código
- Te costará debuggear y mejorar modelos

### P: ¿Hay un test final?

**R**: Sí, implícitamente: **Implementar Q-Learning desde cero** en un ambiente simple. Si puedes hacerlo, estás listo.

---

## 🤝 Contribuir

¿Encontraste un error? ¿Tienes una mejor explicación? ¡Contribuye!

1. Abre un issue en GitHub
2. Propón cambios vía PR
3. Comparte feedback en discusiones

---

## 📝 Notas Finales

**Recuerda**:
- No intentes memorizar todo
- Enfócate en entender conceptos
- La práctica es más importante que la teoría
- Todos estos conceptos se reforzarán durante el estudio de RL

**Cita motivacional**:
> "You don't need to be a mathematician to do RL, but you need to understand the math behind what you're doing."
> — David Silver

---

## 🎯 Tu Progreso

Usa este checklist para trackear tu progreso:

```
Prerequisites
├── [  ] 01_matematicas_basicas.md
│   ├── [  ] Leer documento completo
│   ├── [  ] Completar ejercicios
│   └── [  ] Implementar funciones clave
│
├── [  ] 02_algebra_lineal.md
│   ├── [  ] Leer documento completo
│   ├── [  ] Completar ejercicios
│   └── [  ] Practicar con NumPy
│
├── [  ] 03_calculo_basico.md
│   ├── [  ] Leer documento completo
│   ├── [  ] Completar ejercicios
│   └── [  ] Implementar gradient descent
│
├── [  ] 04_python_numpy.md
│   ├── [  ] Leer documento completo
│   ├── [  ] Completar ejercicios
│   └── [  ] Escribir agente básico
│
└── [  ] 05_conceptos_optimizacion.md
    ├── [  ] Leer documento completo
    ├── [  ] Completar ejercicios
    └── [  ] Comparar optimizadores

Proyecto Final de Prerequisites:
[  ] Implementar Q-Learning tabular en GridWorld
```

---

¡Buena suerte en tu viaje de aprendizaje! 🚀

**Siguiente**: [Matemáticas Básicas](01_matematicas_basicas.md) →
