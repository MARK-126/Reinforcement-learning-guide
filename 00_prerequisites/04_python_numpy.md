# Python y NumPy para Reinforcement Learning

## 🎯 Objetivo

Esta guía cubre **solo lo esencial de Python** que necesitas para entender e implementar algoritmos de RL. No es un curso completo de Python, sino un "fast track" enfocado en RL.

**Prerrequisitos**: Ninguno. Empezamos desde cero.

---

## 1. Python Básico Esencial

### 1.1 Variables y Tipos

```python
# Números
x = 5              # int (entero)
y = 3.14           # float (decimal)
z = 2 + 3j         # complex (complejo, raro en RL)

# Strings (texto)
nombre = "Reinforcement Learning"
inicial = 'R'

# Booleanos
es_episodio_terminado = True
tiene_recompensa = False

# None (valor nulo)
siguiente_estado = None

# Ver tipo
print(type(x))     # <class 'int'>
```

### 1.2 Operaciones Básicas

```python
# Aritméticas
suma = 5 + 3           # 8
resta = 10 - 4         # 6
multiplicacion = 3 * 7 # 21
division = 15 / 4      # 3.75
division_entera = 15 // 4  # 3
modulo = 15 % 4        # 3 (residuo)
potencia = 2 ** 3      # 8

# Comparaciones
5 > 3          # True
10 <= 10       # True
5 == 5         # True (igualdad)
5 != 3         # True (diferente)

# Lógicas
True and False  # False
True or False   # True
not True        # False
```

### 1.3 Estructuras de Datos

#### Listas (Arrays dinámicos)

```python
# Crear lista
estados = [0, 1, 2, 3, 4]
recompensas = [1.5, 2.0, -0.5, 3.2]
mixta = [1, "texto", True, 3.14]

# Acceso (índice empieza en 0)
print(estados[0])      # 0 (primer elemento)
print(estados[-1])     # 4 (último elemento)
print(estados[1:4])    # [1, 2, 3] (slice)

# Modificar
estados[2] = 99        # estados = [0, 1, 99, 3, 4]

# Operaciones
estados.append(5)      # Agregar al final
estados.pop()          # Eliminar último
len(estados)           # Longitud
sum(recompensas)       # Suma de elementos

# Verificar pertenencia
2 in estados           # True
```

**En RL**: Almacenar secuencias de estados, acciones, recompensas.

#### Tuplas (Inmutables)

```python
# Similar a listas pero NO se pueden modificar
transicion = (estado, accion, recompensa, siguiente_estado)
posicion = (3, 5)

# Desempaquetar
s, a, r, s_next = transicion
x, y = posicion

print(x)  # 3
```

**En RL**: Representar experiencias (s, a, r, s', done).

#### Diccionarios (Hash maps)

```python
# Pares clave-valor
Q_values = {
    'estado_0': {'arriba': 0.5, 'abajo': 0.3},
    'estado_1': {'arriba': 0.8, 'abajo': 0.2}
}

# Acceso
print(Q_values['estado_0']['arriba'])  # 0.5

# Agregar
Q_values['estado_2'] = {'arriba': 0.0, 'abajo': 0.0}

# Verificar existencia
if 'estado_0' in Q_values:
    print("Existe")

# Iterar
for estado, acciones in Q_values.items():
    print(f"{estado}: {acciones}")
```

**En RL**: Tablas de Q-values, value functions, políticas.

#### Sets (Conjuntos)

```python
# Colección sin duplicados
estados_visitados = {1, 2, 3, 3, 2, 1}
print(estados_visitados)  # {1, 2, 3}

# Operaciones
estados_visitados.add(4)
estados_visitados.remove(1)
```

### 1.4 Control de Flujo

#### If-Elif-Else

```python
epsilon = 0.1
random_value = 0.05

if random_value < epsilon:
    # Exploración
    accion = random.choice([0, 1, 2, 3])
elif random_value < 0.5:
    # Semi-aleatorio
    accion = policy_semi_random()
else:
    # Explotación
    accion = argmax(Q_values)
```

#### For Loops

```python
# Iterar sobre lista
recompensas = [1, 2, 3, 4, 5]
for r in recompensas:
    print(r)

# Con índice
for i, r in enumerate(recompensas):
    print(f"Recompensa {i}: {r}")

# Range
for episodio in range(1000):  # 0 a 999
    # Entrenar agente
    pass

# Iterar sobre diccionario
for estado, valor in V.items():
    print(f"V({estado}) = {valor}")
```

**En RL**: Episodios de entrenamiento, iteración sobre estados.

#### While Loops

```python
episodio_terminado = False
pasos = 0

while not episodio_terminado and pasos < 200:
    accion = agente.seleccionar_accion(estado)
    siguiente_estado, recompensa, episodio_terminado = env.step(accion)
    pasos += 1
```

### 1.5 Funciones

```python
def calcular_retorno(recompensas, gamma=0.99):
    """
    Calcula retorno descontado.

    Args:
        recompensas: Lista de recompensas [r1, r2, r3, ...]
        gamma: Factor de descuento (default: 0.99)

    Returns:
        float: Retorno total descontado
    """
    retorno = 0
    for t, r in enumerate(recompensas):
        retorno += (gamma ** t) * r
    return retorno

# Uso
recompensas = [1, 1, 1, 10]
G = calcular_retorno(recompensas, gamma=0.9)
print(f"Retorno: {G}")  # 11.629
```

**Funciones lambda** (anónimas):
```python
# Función normal
def cuadrado(x):
    return x**2

# Lambda equivalente
cuadrado = lambda x: x**2

# Uso común en RL
epsilon_decay = lambda episodio: max(0.01, 1.0 * 0.995**episodio)
```

### 1.6 List Comprehensions

```python
# Crear lista con loop tradicional
cuadrados = []
for i in range(10):
    cuadrados.append(i**2)

# List comprehension (más Pythonic)
cuadrados = [i**2 for i in range(10)]

# Con condición
pares = [i for i in range(20) if i % 2 == 0]

# En RL: Filtrar transiciones
buenas_transiciones = [t for t in replay_buffer if t.reward > 0]
```

---

## 2. NumPy Esencial

NumPy es **LA librería** para computación numérica en Python. Absolutamente esencial para RL.

### 2.1 Crear Arrays

```python
import numpy as np

# Desde lista
arr = np.array([1, 2, 3, 4, 5])
matriz = np.array([[1, 2, 3], [4, 5, 6]])

# Arrays especiales
ceros = np.zeros(5)              # [0., 0., 0., 0., 0.]
unos = np.ones((3, 4))           # Matriz 3x4 de unos
identidad = np.eye(5)            # Identidad 5x5
rango = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]

# Aleatorios
random_uniform = np.random.rand(3, 4)      # Uniforme [0,1]
random_normal = np.random.randn(3, 4)      # Normal(0,1)
random_int = np.random.randint(0, 10, 5)   # Enteros [0,10)
```

**En RL**:
```python
# Inicializar Q-table
Q = np.zeros((num_estados, num_acciones))

# Pesos de red neuronal
W = np.random.randn(input_dim, output_dim) * 0.01
```

### 2.2 Propiedades de Arrays

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape       # (2, 3) - dimensiones
arr.ndim        # 2 - número de dimensiones
arr.size        # 6 - total de elementos
arr.dtype       # dtype('int64') - tipo de datos
```

### 2.3 Indexación y Slicing

```python
arr = np.array([10, 20, 30, 40, 50])

# Acceso
arr[0]          # 10
arr[-1]         # 50
arr[1:4]        # array([20, 30, 40])
arr[::2]        # array([10, 30, 50])

# Matrices
matriz = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

matriz[0, 0]    # 1
matriz[1, :]    # array([4, 5, 6]) - fila 1
matriz[:, 2]    # array([3, 6, 9]) - columna 2
matriz[1:, 1:]  # array([[5, 6], [8, 9]])

# Indexación booleana
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
print(arr[mask])  # array([4, 5])
```

**En RL**:
```python
# Obtener Q-values para un estado
Q = np.random.rand(5, 4)  # 5 estados, 4 acciones
estado_actual = 2
q_values_estado_2 = Q[estado_actual, :]

# Filtrar experiencias
recompensas = np.array([1, -1, 0, 5, -2])
positivas = recompensas[recompensas > 0]
```

### 2.4 Operaciones Elemento por Elemento

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Aritméticas (elemento por elemento)
a + b           # array([11, 22, 33, 44])
a * b           # array([10, 40, 90, 160])
a / b           # array([0.1, 0.1, 0.1, 0.1])
a ** 2          # array([1, 4, 9, 16])

# Funciones matemáticas
np.sqrt(a)      # array([1., 1.414, 1.732, 2.])
np.exp(a)       # array([2.718, 7.389, 20.085, 54.598])
np.log(a)       # array([0., 0.693, 1.099, 1.386])
np.abs(a)       # Valor absoluto
np.sin(a)       # Seno
```

**En RL**:
```python
# Actualizar Q-values
Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])

# Descuentos
retornos_descontados = recompensas * (gamma ** np.arange(len(recompensas)))
```

### 2.5 Reducción (Aggregation)

```python
arr = np.array([1, 2, 3, 4, 5])

np.sum(arr)     # 15
np.mean(arr)    # 3.0
np.std(arr)     # 1.414 (desviación estándar)
np.max(arr)     # 5
np.min(arr)     # 1
np.argmax(arr)  # 4 (índice del máximo)
np.argmin(arr)  # 0 (índice del mínimo)

# Para matrices (especificar eje)
matriz = np.array([[1, 2, 3],
                   [4, 5, 6]])

np.sum(matriz)              # 21 (suma total)
np.sum(matriz, axis=0)      # array([5, 7, 9]) - suma columnas
np.sum(matriz, axis=1)      # array([6, 15]) - suma filas
np.max(matriz, axis=1)      # array([3, 6]) - máximo por fila
```

**En RL**:
```python
# Política greedy
mejor_accion = np.argmax(Q[estado, :])

# Recompensa promedio
recompensa_media = np.mean(recompensas_episodio)

# Value iteration: max sobre acciones
V_new = np.max(Q, axis=1)
```

### 2.6 Broadcasting

NumPy automáticamente expande arrays para operaciones:

```python
# Escalar + array
arr = np.array([1, 2, 3])
resultado = arr + 10  # array([11, 12, 13])

# Vector + matriz (broadcasting)
matriz = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([10, 20, 30])

resultado = matriz + vector
# array([[11, 22, 33],
#        [14, 25, 36]])

# Cada fila de matriz se suma con vector
```

**En RL**:
```python
# Aplicar descuento a todos los valores
V_descontado = gamma * V

# Normalizar probabilidades
probs = np.array([0.2, 0.5, 0.3])
probs_normalized = probs / np.sum(probs)
```

### 2.7 Álgebra Lineal

```python
# Producto punto
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # 32

# Multiplicación matriz-vector
A = np.array([[1, 2], [3, 4]])
v = np.array([5, 6])
resultado = A @ v  # array([17, 39])
# También: np.dot(A, v)

# Multiplicación matriz-matriz
B = np.array([[7, 8], [9, 10]])
C = A @ B

# Transpuesta
A_T = A.T

# Inversa
A_inv = np.linalg.inv(A)

# Norma
norma = np.linalg.norm(v)
```

**En RL - Value Iteration matricial**:
```python
# V = (I - gamma*P)^(-1) * R
I = np.eye(n_states)
P = transition_matrix
R = reward_vector
gamma = 0.9

V = np.linalg.inv(I - gamma * P) @ R
```

### 2.8 Funciones Útiles para RL

#### Softmax

```python
def softmax(x):
    """Convierte logits en probabilidades"""
    exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
    return exp_x / np.sum(exp_x)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(probs)  # array([0.659, 0.242, 0.099])
```

#### Epsilon-greedy

```python
def epsilon_greedy(Q, estado, epsilon=0.1):
    """Selecciona acción con epsilon-greedy"""
    if np.random.random() < epsilon:
        # Exploración
        return np.random.randint(len(Q[estado]))
    else:
        # Explotación
        return np.argmax(Q[estado])
```

#### One-hot encoding

```python
def one_hot(index, size):
    """Crea vector one-hot"""
    vec = np.zeros(size)
    vec[index] = 1
    return vec

# Ejemplo: estado 2 de 5 posibles
estado_encoded = one_hot(2, 5)
print(estado_encoded)  # [0. 0. 1. 0. 0.]
```

---

## 3. Clases y Programación Orientada a Objetos

### 3.1 Definir Clase

```python
class QLearningAgent:
    """Agente Q-Learning"""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """Constructor"""
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Inicializar Q-table
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        """Selecciona acción usando epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state, :])

    def update(self, state, action, reward, next_state, done):
        """Actualiza Q-value"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state, :])

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

# Uso
agente = QLearningAgent(n_states=16, n_actions=4)
accion = agente.select_action(estado=5)
agente.update(5, accion, 1.0, 6, False)
```

### 3.2 Herencia

```python
class Agent:
    """Clase base para agentes"""
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def select_action(self, state):
        raise NotImplementedError

class RandomAgent(Agent):
    """Agente aleatorio"""
    def select_action(self, state):
        return np.random.randint(self.n_actions)

class GreedyAgent(Agent):
    """Agente greedy"""
    def __init__(self, n_actions, Q):
        super().__init__(n_actions)
        self.Q = Q

    def select_action(self, state):
        return np.argmax(self.Q[state, :])
```

---

## 4. Librerías Esenciales para RL

### 4.1 Matplotlib (Visualización)

```python
import matplotlib.pyplot as plt

# Gráfico simple
episodios = range(100)
recompensas = np.random.randn(100).cumsum()

plt.plot(episodios, recompensas)
plt.xlabel('Episodio')
plt.ylabel('Recompensa Acumulada')
plt.title('Progreso del Entrenamiento')
plt.grid()
plt.show()

# Múltiples líneas
plt.plot(episodios, recompensas, label='Agente 1')
plt.plot(episodios, recompensas * 1.2, label='Agente 2')
plt.legend()
plt.show()

# Heatmap (Q-table)
plt.imshow(Q, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Acciones')
plt.ylabel('Estados')
plt.title('Q-Table')
plt.show()
```

### 4.2 Gymnasium (Ambientes RL)

```python
import gymnasium as gym

# Crear ambiente
env = gym.make('CartPole-v1')

# Reset
estado, info = env.reset()

# Loop de interacción
done = False
while not done:
    # Seleccionar acción
    accion = env.action_space.sample()  # Aleatoria

    # Ejecutar
    siguiente_estado, recompensa, terminated, truncated, info = env.step(accion)
    done = terminated or truncated

    estado = siguiente_estado

env.close()
```

### 4.3 Collections (Estructuras útiles)

```python
from collections import deque, defaultdict, Counter

# Deque (cola de doble extremo) - para replay buffer
replay_buffer = deque(maxlen=10000)
replay_buffer.append((s, a, r, s_next, done))

# DefaultDict - Q-table con inicialización automática
Q = defaultdict(lambda: np.zeros(n_actions))
Q[estado][accion] += alpha * td_error

# Counter - contar visitas
visitas = Counter()
visitas[estado] += 1
```

---

## 5. Tips y Mejores Prácticas

### 5.1 Debugging

```python
# Print informativo
print(f"Estado: {estado}, Acción: {accion}, Recompensa: {recompensa:.2f}")

# Verificar shapes
print(f"Shape de Q: {Q.shape}")
print(f"Shape de estado: {estado.shape}")

# Aserciones
assert Q.shape == (n_states, n_actions), "Q-table tiene tamaño incorrecto"
assert 0 <= epsilon <= 1, "Epsilon debe estar en [0, 1]"

# Verificar valores
print(f"Min Q: {Q.min():.2f}, Max Q: {Q.max():.2f}, Mean Q: {Q.mean():.2f}")
```

### 5.2 Manejo de Random Seeds

```python
import random
import numpy as np

def set_seed(seed=42):
    """Fija seeds para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    # Para PyTorch:
    # import torch
    # torch.manual_seed(seed)

set_seed(42)  # Resultados reproducibles
```

### 5.3 Guardar y Cargar

```python
# NumPy arrays
np.save('Q_table.npy', Q)
Q_loaded = np.load('Q_table.npy')

# Múltiples arrays
np.savez('checkpoint.npz', Q=Q, V=V, policy=policy)
data = np.load('checkpoint.npz')
Q = data['Q']

# JSON (para hyperparameters)
import json

config = {
    'alpha': 0.1,
    'gamma': 0.99,
    'epsilon': 0.1,
    'n_episodes': 1000
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

with open('config.json', 'r') as f:
    config_loaded = json.load(f)
```

---

## 6. Ejercicios Prácticos

### Ejercicio 1: Implementar Epsilon Decay

```python
def epsilon_decay(episodio, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.995):
    """
    Implementa decaimiento exponencial de epsilon

    Args:
        episodio: Número de episodio actual
        epsilon_start: Epsilon inicial
        epsilon_end: Epsilon mínimo
        decay_rate: Tasa de decaimiento

    Returns:
        Epsilon actual
    """
    # TU CÓDIGO AQUÍ
    pass

# Test
for ep in [0, 100, 500, 1000]:
    print(f"Episodio {ep}: ε = {epsilon_decay(ep):.3f}")
```

<details>
<summary>Ver solución</summary>

```python
def epsilon_decay(episodio, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.995):
    epsilon = max(epsilon_end, epsilon_start * (decay_rate ** episodio))
    return epsilon
```
</details>

### Ejercicio 2: Calcular TD Error

```python
def compute_td_error(Q, state, action, reward, next_state, done, gamma=0.99):
    """
    Calcula TD error: δ = r + γ*max_a Q(s',a) - Q(s,a)

    Returns:
        float: TD error
    """
    # TU CÓDIGO AQUÍ
    pass
```

<details>
<summary>Ver solución</summary>

```python
def compute_td_error(Q, state, action, reward, next_state, done, gamma=0.99):
    if done:
        target = reward
    else:
        target = reward + gamma * np.max(Q[next_state, :])

    td_error = target - Q[state, action]
    return td_error
```
</details>

### Ejercicio 3: Replay Buffer

```python
class ReplayBuffer:
    """Replay buffer para almacenar experiencias"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Agrega experiencia"""
        # TU CÓDIGO AQUÍ
        pass

    def sample(self, batch_size):
        """Muestrea batch aleatorio"""
        # TU CÓDIGO AQUÍ
        pass

    def __len__(self):
        return len(self.buffer)
```

<details>
<summary>Ver solución</summary>

```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)
```
</details>

---

## 7. Cheat Sheet

### Python Básico
```python
# Listas
lista = [1, 2, 3]
lista.append(4)
len(lista)

# Diccionarios
d = {'key': 'value'}
d['key']

# Loops
for i in range(10): pass
while condition: pass

# Funciones
def func(x, y=0): return x + y

# Clases
class Clase:
    def __init__(self): pass
```

### NumPy
```python
# Crear
np.array([1, 2, 3])
np.zeros((3, 4))
np.random.rand(3, 4)

# Operaciones
arr.sum(), arr.mean(), arr.std()
np.max(arr), np.argmax(arr)
np.dot(a, b), A @ B

# Indexación
arr[0], arr[:, 1]
arr[arr > 0]
```

---

## 8. Recursos Adicionales

### Tutoriales Python
- [Python.org Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)
- [W3Schools Python](https://www.w3schools.com/python/)

### NumPy
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

### RL Específico
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)

---

## 9. Autoevaluación

¿Puedes...?

- [ ] Crear y manipular listas, diccionarios
- [ ] Escribir loops y funciones
- [ ] Crear arrays de NumPy
- [ ] Usar indexación y slicing
- [ ] Aplicar operaciones matriciales
- [ ] Definir clases simples

Si respondiste todo, ¡perfecto! Continúa con [Optimización](05_conceptos_optimizacion.md).

---

## Próximos Pasos

1. **[Optimización](05_conceptos_optimizacion.md)** - Algoritmos de optimización
2. **[Fundamentos de RL](../01_fundamentos/introduccion.md)** - ¡Empezar con RL!

¡Excelente progreso! 🚀
