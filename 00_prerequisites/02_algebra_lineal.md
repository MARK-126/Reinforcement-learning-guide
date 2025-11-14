# Álgebra Lineal para Reinforcement Learning

## 🎯 Por Qué Necesitas Álgebra Lineal en RL

En Reinforcement Learning, trabajamos constantemente con:
- **Vectores**: Para representar estados (posición, velocidad, etc.)
- **Matrices**: Para transiciones de probabilidad, redes neuronales
- **Operaciones matriciales**: Para calcular valores, propagar gradientes

Esta guía te enseña **solo lo esencial** para RL, desde cero.

---

## 1. Vectores desde Cero

### 1.1 ¿Qué es un Vector?

Un **vector** es una lista ordenada de números. Piénsalo como:
- Una flecha en el espacio (tiene dirección y magnitud)
- Coordenadas de una ubicación
- Características de un estado

**Ejemplos**:
```
Posición 2D: v = [3, 5]
              ↑  ↑
              x  y

Estado CartPole: s = [x, ẋ, θ, θ̇]
                      ↑  ↑  ↑  ↑
                      posición, velocidad, ángulo, velocidad_angular

Vector de recompensas: r = [1.5, 2.0, -0.5, 3.2]
```

### 1.2 Notación

**Matemática**:
```
v = [v₁, v₂, v₃]  o  v = ⎡v₁⎤
                         ⎢v₂⎥
                         ⎣v₃⎦
```

**Python/NumPy**:
```python
import numpy as np

v = np.array([3, 5])           # Vector 2D
s = np.array([0.1, 0.2, 0.3])  # Vector 3D
```

### 1.3 Operaciones con Vectores

#### Suma de Vectores

Suma elemento por elemento:
```
[a₁]   [b₁]   [a₁ + b₁]
[a₂] + [b₂] = [a₂ + b₂]
[a₃]   [b₃]   [a₃ + b₃]
```

**Ejemplo**:
```
v₁ = [1, 2, 3]
v₂ = [4, 5, 6]
v₁ + v₂ = [1+4, 2+5, 3+6] = [5, 7, 9]
```

**Python**:
```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
suma = v1 + v2  # array([5, 7, 9])
```

#### Multiplicación por Escalar

Multiplica cada elemento por un número:
```
c · [a₁]   [c·a₁]
    [a₂] = [c·a₂]
    [a₃]   [c·a₃]
```

**Ejemplo**:
```
2 · [1, 2, 3] = [2, 4, 6]
```

**Python**:
```python
v = np.array([1, 2, 3])
resultado = 2 * v  # array([2, 4, 6])
```

**En RL**: Descontar recompensas: `γ * V(s')`

#### Producto Punto (Dot Product)

Multiplica elementos correspondientes y suma:
```
v₁ · v₂ = v₁[0]·v₂[0] + v₁[1]·v₂[1] + ... + v₁[n]·v₂[n]
```

**Ejemplo**:
```
v₁ = [1, 2, 3]
v₂ = [4, 5, 6]
v₁ · v₂ = 1·4 + 2·5 + 3·6 = 4 + 10 + 18 = 32
```

**Python**:
```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)  # 32
```

**En RL**:
- Calcular Q-values: `Q = w · φ(s,a)` (aproximación lineal)
- Producto de características y pesos en redes neuronales

#### Norma (Magnitud)

La **norma** ||v|| es la "longitud" del vector:

```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

**Ejemplo**:
```
v = [3, 4]
||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5
```

**Python**:
```python
v = np.array([3, 4])
norma = np.linalg.norm(v)  # 5.0
```

**En RL**: Medir magnitud de gradientes, distancia entre estados

---

## 2. Matrices desde Cero

### 2.1 ¿Qué es una Matriz?

Una **matriz** es una tabla rectangular de números. Piénsala como:
- Colección de vectores
- Tabla de transiciones de probabilidad
- Pesos de una capa de red neuronal

**Notación**:
```
      m columnas
    ┌──────────┐
n   │ a₁₁ a₁₂ │
f   │ a₂₁ a₂₂ │  Matriz A de tamaño n × m
i   │ a₃₁ a₃₂ │
l   └──────────┘
a
s
```

**Ejemplo - Tabla de transiciones**:
```
     s₁  s₂  s₃
s₁ [ 0.1 0.7 0.2 ]
s₂ [ 0.3 0.3 0.4 ]
s₃ [ 0.0 0.5 0.5 ]

P[i][j] = probabilidad de ir de estado i a estado j
```

**Python**:
```python
# Matriz 3×3
A = np.array([
    [0.1, 0.7, 0.2],
    [0.3, 0.3, 0.4],
    [0.0, 0.5, 0.5]
])

print(A.shape)  # (3, 3)
print(A[0, 1])  # 0.7 (fila 0, columna 1)
```

### 2.2 Tipos de Matrices Especiales

#### Matriz Identidad (I)

Tiene 1s en la diagonal, 0s en el resto:
```
I₃ = [ 1  0  0 ]
     [ 0  1  0 ]
     [ 0  0  1 ]
```

**Propiedad**: A · I = I · A = A

**Python**:
```python
I = np.eye(3)  # Matriz identidad 3×3
```

**En RL**: Aparece en ecuaciones de Bellman en forma matricial: `V = (I - γP)⁻¹R`

#### Matriz Diagonal

Solo tiene valores distintos de cero en la diagonal:
```
D = [ 2  0  0 ]
    [ 0  5  0 ]
    [ 0  0  3 ]
```

**Python**:
```python
D = np.diag([2, 5, 3])
```

#### Matriz Transpuesta

La transpuesta Aᵀ intercambia filas por columnas:
```
     [ 1  2  3 ]        [ 1  4 ]
A =  [ 4  5  6 ]   Aᵀ = [ 2  5 ]
                         [ 3  6 ]
```

**Python**:
```python
A = np.array([[1, 2, 3], [4, 5, 6]])
A_transpuesta = A.T
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
```

### 2.3 Operaciones con Matrices

#### Suma de Matrices

Elemento por elemento (deben tener mismo tamaño):
```
[ 1  2 ]   [ 5  6 ]   [ 6   8 ]
[ 3  4 ] + [ 7  8 ] = [ 10  12 ]
```

**Python**:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B
# array([[ 6,  8],
#        [10, 12]])
```

#### Multiplicación Matriz-Vector

Multiplica matriz A (n×m) por vector v (m×1) para obtener vector resultado (n×1):

```
[ a₁₁  a₁₂ ]   [ v₁ ]   [ a₁₁·v₁ + a₁₂·v₂ ]
[ a₂₁  a₂₂ ] · [ v₂ ] = [ a₂₁·v₁ + a₂₂·v₂ ]
```

**Ejemplo**:
```
[ 1  2 ]   [ 5 ]   [ 1·5 + 2·6 ]   [ 17 ]
[ 3  4 ] · [ 6 ] = [ 3·5 + 4·6 ] = [ 39 ]
```

**Python**:
```python
A = np.array([[1, 2], [3, 4]])
v = np.array([5, 6])
resultado = A @ v  # array([17, 39])
# También: np.dot(A, v)
```

**En RL**:
- Calcular valores: `V_new = P · V_old`
- Propagación en redes neuronales: `h = W · x`

#### Multiplicación Matriz-Matriz

Para multiplicar A (n×m) por B (m×p), resultado es C (n×p):

```
Cᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ
```

**Ejemplo visual**:
```
[ 1  2 ]   [ 5  6 ]   [ 1·5+2·7  1·6+2·8 ]   [ 19  22 ]
[ 3  4 ] · [ 7  8 ] = [ 3·5+4·7  3·6+4·8 ] = [ 43  50 ]
```

**Python**:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # array([[19, 22], [43, 50]])
```

**⚠️ IMPORTANTE**: El orden importa: A·B ≠ B·A (generalmente)

### 2.4 Inversa de una Matriz

La **inversa** A⁻¹ satisface: A · A⁻¹ = A⁻¹ · A = I

**Ejemplo**:
```
     [ 4  7 ]              [ 0.6  -0.7 ]
A =  [ 2  6 ]    A⁻¹ =     [ -0.2  0.4 ]

Verifica: A · A⁻¹ = I
```

**Python**:
```python
A = np.array([[4, 7], [2, 6]])
A_inv = np.linalg.inv(A)
print(A @ A_inv)  # Aproximadamente [[1, 0], [0, 1]]
```

**En RL**:
- Resolver Bellman directamente: `V = (I - γP)⁻¹ · R`
- Solo para MDPs pequeños (computacionalmente costoso)

---

## 3. Aplicaciones Directas en RL

### 3.1 Representación de Estados como Vectores

**CartPole**:
```python
estado = np.array([
    0.02,   # posición del carro
    0.15,   # velocidad del carro
    -0.05,  # ángulo del poste
    0.20    # velocidad angular
])
```

### 3.2 Value Function como Vector

Para n estados, V es un vector de n elementos:
```python
# GridWorld 3×3 = 9 estados
V = np.array([0.0, -1.0, -2.0, -1.0, -2.0, -3.0, -2.0, -3.0, 0.0])
#              s₁    s₂    s₃    s₄    s₅    s₆    s₇    s₈    s₉
```

### 3.3 Matriz de Transición P

Para ambiente con n estados:
```python
# P[i][j] = probabilidad de transición s_i → s_j
P = np.array([
    [0.1, 0.7, 0.2],  # Desde estado 0
    [0.3, 0.3, 0.4],  # Desde estado 1
    [0.0, 0.5, 0.5]   # Desde estado 2
])

# Calcular siguiente distribución de estados
estado_actual = np.array([1, 0, 0])  # En estado 0
siguiente_dist = P.T @ estado_actual
# array([0.1, 0.7, 0.2]) - probabilidades de estar en cada estado
```

### 3.4 Ecuación de Bellman en Forma Matricial

La ecuación de Bellman:
```
V^π(s) = Σₐ π(a|s) Σₛ′ P(s′|s,a)[R(s,a,s′) + γV^π(s′)]
```

Se puede escribir como:
```
V = R + γPV
```

Donde:
- **V**: vector de valores (n × 1)
- **R**: vector de recompensas (n × 1)
- **P**: matriz de transiciones (n × n)
- **γ**: escalar de descuento

**Solución directa**:
```
V = (I - γP)⁻¹ R
```

**Código completo**:
```python
import numpy as np

# Definir MDP simple
gamma = 0.9
R = np.array([0, 0, 1])  # Recompensas
P = np.array([           # Transiciones
    [0.1, 0.7, 0.2],
    [0.3, 0.3, 0.4],
    [0.0, 0.5, 0.5]
])

# Resolver Bellman
I = np.eye(3)
V = np.linalg.inv(I - gamma * P) @ R
print("Valores óptimos:", V)
# V ≈ [1.92, 2.62, 10.0]
```

### 3.5 Redes Neuronales (Adelanto)

Una capa de red neuronal es simplemente:
```
h = activation(W · x + b)
```

Donde:
- **W**: matriz de pesos
- **x**: vector de entrada (estado)
- **b**: vector de bias
- **h**: vector de salida (valores Q, etc.)

**Ejemplo**:
```python
# Capa simple: 4 entradas → 2 salidas
W = np.random.randn(2, 4)  # Pesos
b = np.zeros(2)             # Bias
x = np.array([1, 2, 3, 4])  # Estado

# Forward pass
z = W @ x + b               # Combinación lineal
h = np.maximum(0, z)        # ReLU activation
print("Salida:", h)
```

---

## 4. Operaciones Útiles en NumPy

### 4.1 Creación de Arrays

```python
import numpy as np

# Vectores
v = np.array([1, 2, 3])
ceros = np.zeros(5)              # [0, 0, 0, 0, 0]
unos = np.ones(3)                # [1, 1, 1]
rango = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
random = np.random.randn(4)      # 4 números aleatorios N(0,1)

# Matrices
M = np.array([[1, 2], [3, 4]])
ceros_mat = np.zeros((3, 4))     # Matriz 3×4 de ceros
identidad = np.eye(5)            # Identidad 5×5
random_mat = np.random.rand(2, 3) # 2×3 aleatoria [0,1]
```

### 4.2 Indexación y Slicing

```python
v = np.array([10, 20, 30, 40, 50])

v[0]      # 10 (primer elemento)
v[-1]     # 50 (último elemento)
v[1:4]    # array([20, 30, 40]) (slice)
v[::2]    # array([10, 30, 50]) (cada 2 elementos)

# Matrices
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

M[0, 0]   # 1 (fila 0, columna 0)
M[1, :]   # array([4, 5, 6]) (toda la fila 1)
M[:, 2]   # array([3, 6, 9]) (toda la columna 2)
M[1:, 1:] # array([[5, 6], [8, 9]]) (submatriz)
```

### 4.3 Broadcasting

NumPy automáticamente "expande" arrays de diferente tamaño:

```python
v = np.array([1, 2, 3])
M = np.array([[10],
              [20],
              [30]])

# Broadcasting: suma cada fila de M con v
resultado = M + v
# array([[11, 12, 13],
#        [21, 22, 23],
#        [31, 32, 33]])
```

**En RL**: Útil para aplicar operaciones a lotes de estados.

### 4.4 Reducción

```python
v = np.array([1, 2, 3, 4, 5])

np.sum(v)        # 15 (suma total)
np.mean(v)       # 3.0 (promedio)
np.max(v)        # 5 (máximo)
np.argmax(v)     # 4 (índice del máximo)

# Para matrices
M = np.array([[1, 2, 3],
              [4, 5, 6]])

np.sum(M, axis=0)  # array([5, 7, 9]) - suma por columnas
np.sum(M, axis=1)  # array([ 6, 15])  - suma por filas
np.max(M, axis=1)  # array([3, 6])    - máximo por fila
```

---

## 5. Ejercicios Prácticos

### Ejercicio 1: Producto Punto
Calcula v₁ · v₂ manualmente y verifica con NumPy:
```
v₁ = [2, 3, 4]
v₂ = [1, 0, 2]
```

<details>
<summary>Ver solución</summary>

```python
# Manual
resultado = 2*1 + 3*0 + 4*2 = 2 + 0 + 8 = 10

# NumPy
v1 = np.array([2, 3, 4])
v2 = np.array([1, 0, 2])
print(np.dot(v1, v2))  # 10
```
</details>

### Ejercicio 2: Multiplicación Matriz-Vector
Multiplica manualmente:
```
A = [ 1  2 ]    v = [ 3 ]
    [ 4  5 ]        [ 6 ]
```

<details>
<summary>Ver solución</summary>

```python
# Manual
fila 1: 1*3 + 2*6 = 3 + 12 = 15
fila 2: 4*3 + 5*6 = 12 + 30 = 42

resultado = [15, 42]

# NumPy
A = np.array([[1, 2], [4, 5]])
v = np.array([3, 6])
print(A @ v)  # array([15, 42])
```
</details>

### Ejercicio 3: Ecuación de Bellman
Dado γ=0.5, R=[1, 0, 10], y P=I (identidad), resuelve V = R + γPV.

<details>
<summary>Ver solución</summary>

```python
import numpy as np

gamma = 0.5
R = np.array([1, 0, 10])
P = np.eye(3)

# V = R + γPV
# V - γPV = R
# (I - γP)V = R
# V = (I - γP)⁻¹ R

I = np.eye(3)
V = np.linalg.inv(I - gamma * P) @ R
print(V)  # array([ 2.,  0., 20.])
```

**Interpretación**: Con γ=0.5 y transiciones a mismo estado, V = R/(1-γ) = 2R.
</details>

### Ejercicio 4: Encontrar Mejor Acción
Dado Q-values para 3 acciones, encuentra la mejor acción:
```python
Q = np.array([0.2, 0.8, 0.5])
```

<details>
<summary>Ver solución</summary>

```python
Q = np.array([0.2, 0.8, 0.5])
mejor_accion = np.argmax(Q)
print(f"Mejor acción: {mejor_accion}")  # 1
print(f"Valor Q: {Q[mejor_accion]}")     # 0.8
```
</details>

---

## 6. Conceptos Avanzados (Opcional)

### 6.1 Eigenvalues y Eigenvectors

Un **eigenvector** v de matriz A satisface:
```
A · v = λ · v
```
Donde λ (lambda) es el **eigenvalue**.

**En RL**:
- Analizar convergencia de algoritmos iterativos
- Matriz de transición P tiene eigenvalue dominante relacionado con tasa de convergencia

```python
A = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)    # [3. 1.]
print("Eigenvectors:\n", eigenvectors)
```

### 6.2 Descomposición SVD

Singular Value Decomposition descompone matriz en:
```
A = U · Σ · Vᵀ
```

**En Deep RL**: Comprimir representaciones, análisis de redes neuronales.

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, VT = np.linalg.svd(A)
print("Valores singulares:", S)
```

---

## 7. Cheat Sheet: Operaciones Esenciales

| Operación | Símbolo | NumPy | Uso en RL |
|-----------|---------|-------|-----------|
| Suma vectores | v₁ + v₂ | `v1 + v2` | Actualizar estados |
| Producto escalar | c·v | `c * v` | Descuento: γ·V |
| Producto punto | v₁·v₂ | `np.dot(v1,v2)` | Q = w·φ |
| Norma | \|\|v\|\| | `np.linalg.norm(v)` | Magnitud gradiente |
| Mult. matriz-vector | A·v | `A @ v` | V_new = P·V |
| Mult. matriz-matriz | A·B | `A @ B` | Composición |
| Transpuesta | Aᵀ | `A.T` | Cambiar dimensiones |
| Inversa | A⁻¹ | `np.linalg.inv(A)` | Bellman directo |
| Identidad | I | `np.eye(n)` | (I-γP) |
| Max | max(v) | `np.max(v)` | Bellman optimality |
| Argmax | argmax(v) | `np.argmax(v)` | Política greedy |

---

## 8. Recursos Adicionales

### Videos (Español)
- [Khan Academy - Álgebra Lineal](https://es.khanacademy.org/math/linear-algebra)
- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (inglés, subtítulos)

### Libros
- Strang, Gilbert. "Introduction to Linear Algebra"
- [Deep Learning Book - Math Appendix](https://www.deeplearningbook.org/contents/linear_algebra.html)

### Práctica
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

---

## 9. Autoevaluación

¿Puedes responder estas preguntas?

- [ ] ¿Qué es un producto punto y cómo se calcula?
- [ ] ¿Cómo se multiplica una matriz por un vector?
- [ ] ¿Qué hace np.argmax()?
- [ ] ¿Para qué sirve la matriz identidad?
- [ ] ¿Cómo se escribe la ecuación de Bellman en forma matricial?

Si respondiste todo, ¡excelente! Continúa con [Cálculo Básico](03_calculo_basico.md).

---

## Próximos Pasos

1. **[Cálculo Básico](03_calculo_basico.md)** - Derivadas y gradientes para Deep RL
2. **[Python y NumPy](04_python_numpy.md)** - Programación práctica
3. **[Optimización](05_conceptos_optimizacion.md)** - Gradient descent y más

¡Sigue adelante! 🚀
