# Cálculo Básico para Reinforcement Learning

## 🎯 Por Qué Necesitas Cálculo en Deep RL

En Deep Reinforcement Learning con redes neuronales, necesitas:
- **Derivadas**: Para saber cómo cambiar parámetros
- **Gradientes**: Dirección de máximo cambio
- **Chain Rule**: Backpropagation en redes neuronales
- **Gradient Descent**: Optimización de políticas y value functions

Esta guía cubre **solo lo esencial** para Deep RL, desde cero.

---

## 1. Derivadas desde Cero

### 1.1 ¿Qué es una Derivada?

La **derivada** mide **qué tan rápido cambia** una función.

**Analogía**: Si conduces un carro:
- **Posición** x(t): dónde estás
- **Velocidad** v(t) = dx/dt: qué tan rápido cambias de posición
- **Aceleración** a(t) = dv/dt: qué tan rápido cambia tu velocidad

### 1.2 Definición Intuitiva

La derivada de f(x) en punto x es:
```
f'(x) = lim[h→0] [f(x+h) - f(x)] / h
```

**Interpretación geométrica**: Pendiente de la recta tangente.

**Ejemplo visual**:
```
   f(x) = x²

    |
  9 |       •  (3, 9)
    |      /
  4 |  •  /     Pendiente en x=2: f'(2) = 4
    | /  •
  1 |•  (2, 4)
    |________
    0  1  2  3

En x=2, la función está creciendo con pendiente 4
```

### 1.3 Notaciones de Derivada

Todas significan lo mismo:
```
f'(x)     ("f prima de x")
df/dx     ("derivada de f respecto a x")
∂f/∂x     ("derivada parcial" - veremos después)
```

### 1.4 Reglas Básicas de Derivación

#### Constantes
```
f(x) = c
f'(x) = 0

Ejemplo: f(x) = 5  →  f'(x) = 0
```

#### Potencias (Regla del Poder)
```
f(x) = xⁿ
f'(x) = n·xⁿ⁻¹

Ejemplos:
f(x) = x²   →  f'(x) = 2x
f(x) = x³   →  f'(x) = 3x²
f(x) = x    →  f'(x) = 1
f(x) = √x   →  f'(x) = 1/(2√x)
```

**Verificación con Python**:
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def f_prima(x):
    return 2*x

x = np.linspace(-3, 3, 100)
plt.plot(x, f(x), label='f(x) = x²')
plt.plot(x, f_prima(x), label="f'(x) = 2x")
plt.legend()
plt.grid()
plt.show()
```

#### Suma y Resta
```
f(x) = g(x) + h(x)
f'(x) = g'(x) + h'(x)

Ejemplo:
f(x) = 3x² + 5x
f'(x) = 6x + 5
```

#### Multiplicación por Constante
```
f(x) = c·g(x)
f'(x) = c·g'(x)

Ejemplo:
f(x) = 5x³
f'(x) = 5·3x² = 15x²
```

#### Regla del Producto
```
f(x) = g(x)·h(x)
f'(x) = g'(x)·h(x) + g(x)·h'(x)

Ejemplo:
f(x) = x²·sin(x)
f'(x) = 2x·sin(x) + x²·cos(x)
```

#### Regla de la Cadena (Chain Rule) ⭐
**La más importante para Deep Learning**

```
f(x) = g(h(x))
f'(x) = g'(h(x))·h'(x)

"Derivada de la función exterior evaluada en la interior,
multiplicada por la derivada de la interior"
```

**Ejemplo simple**:
```
f(x) = (x² + 1)³

Sea u = x² + 1, entonces f = u³

f'(x) = d/du[u³]·d/dx[x² + 1]
      = 3u²·2x
      = 3(x² + 1)²·2x
      = 6x(x² + 1)²
```

**En Python (aproximación numérica)**:
```python
def derivada_numerica(f, x, h=1e-5):
    """Aproxima f'(x) usando diferencias finitas"""
    return (f(x + h) - f(x)) / h

f = lambda x: (x**2 + 1)**3
x = 2.0

# Derivada numérica
aprox = derivada_numerica(f, x)

# Derivada analítica: 6x(x² + 1)²
exacta = 6*x*(x**2 + 1)**2

print(f"Aproximada: {aprox:.6f}")
print(f"Exacta:     {exacta:.6f}")
```

---

## 2. Funciones Importantes en RL

### 2.1 Función Exponencial

```
f(x) = eˣ
f'(x) = eˣ

¡La derivada es ella misma!
```

**En RL**: Softmax, Boltzmann exploration

**Python**:
```python
import numpy as np

x = np.array([0, 1, 2, -1])
y = np.exp(x)  # array([1.   , 2.718, 7.389, 0.368])
```

### 2.2 Función Logarítmica

```
f(x) = ln(x)   (logaritmo natural)
f'(x) = 1/x

f(x) = log(x)  (puede ser cualquier base)
```

**En RL**: Log-probabilities en policy gradient, entropía

```python
x = np.array([1, 2, np.e, 10])
y = np.log(x)  # array([0.   , 0.693, 1.   , 2.303])
```

### 2.3 Funciones de Activación

#### Sigmoid (σ)
```
σ(x) = 1 / (1 + e⁻ˣ)

Rango: (0, 1)
σ'(x) = σ(x)·(1 - σ(x))
```

**Propiedad útil**: La derivada se expresa en términos de la función misma.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

x = np.linspace(-6, 6, 100)
plt.plot(x, sigmoid(x), label='σ(x)')
plt.plot(x, sigmoid_derivative(x), label="σ'(x)")
plt.legend()
plt.grid()
plt.show()
```

**En RL**: Output de probabilidades, gates en LSTMs

#### Tanh
```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

Rango: (-1, 1)
tanh'(x) = 1 - tanh²(x)
```

```python
def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t**2
```

**En RL**: Activación en actor networks (acciones continuas)

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x) = { x  si x > 0
                      { 0  si x ≤ 0

ReLU'(x) = { 1  si x > 0
           { 0  si x ≤ 0
```

**Más utilizada en Deep RL moderno**

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

x = np.linspace(-3, 3, 100)
plt.plot(x, relu(x), label='ReLU(x)')
plt.plot(x, relu_derivative(x), label="ReLU'(x)")
plt.legend()
plt.grid()
plt.show()
```

**Ventajas**: No vanishing gradient, computacionalmente eficiente

#### Softmax
```
softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ

Convierte vector en distribución de probabilidad
```

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
    return exp_x / np.sum(exp_x)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(probs)  # array([0.659, 0.242, 0.099])
print(np.sum(probs))  # 1.0
```

**En RL**: Convertir Q-values en política estocástica

---

## 3. Derivadas Parciales

### 3.1 ¿Qué son?

Cuando una función depende de **múltiples variables**, la **derivada parcial** mide cómo cambia respecto a UNA variable, manteniendo las demás constantes.

**Notación**:
```
f(x, y)
∂f/∂x   - derivada parcial respecto a x (y es constante)
∂f/∂y   - derivada parcial respecto a y (x es constante)
```

### 3.2 Ejemplo: Función de Pérdida

```
f(w₁, w₂) = w₁² + w₂² + 3w₁w₂

∂f/∂w₁ = 2w₁ + 3w₂     (tratar w₂ como constante)
∂f/∂w₂ = 2w₂ + 3w₁     (tratar w₁ como constante)
```

**Interpretación**: Si estás en punto (w₁, w₂), las derivadas parciales te dicen cómo cambia f si te mueves solo en dirección w₁ o solo en dirección w₂.

**Python**:
```python
def f(w1, w2):
    return w1**2 + w2**2 + 3*w1*w2

def df_dw1(w1, w2):
    return 2*w1 + 3*w2

def df_dw2(w1, w2):
    return 2*w2 + 3*w1

# Evaluar en punto (1, 2)
w1, w2 = 1, 2
print(f"f({w1}, {w2}) = {f(w1, w2)}")           # 11
print(f"∂f/∂w₁ = {df_dw1(w1, w2)}")             # 8
print(f"∂f/∂w₂ = {df_dw2(w1, w2)}")             # 7
```

---

## 4. Gradientes

### 4.1 Definición

El **gradiente** ∇f es un **vector** de todas las derivadas parciales:

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Ejemplo**:
```
f(w₁, w₂) = w₁² + w₂² + 3w₁w₂

∇f = [∂f/∂w₁, ∂f/∂w₂] = [2w₁ + 3w₂, 2w₂ + 3w₁]
```

### 4.2 Interpretación Geométrica

El gradiente **apunta en la dirección de mayor crecimiento** de la función.

```
    ↗ ∇f     (dirección de subida más rápida)
   /
  • (punto actual)
   \
    ↘ -∇f    (dirección de bajada más rápida)
```

**En RL**: Para **maximizar** recompensa, seguimos +∇J (gradient ascent)
**En supervised learning**: Para **minimizar** loss, seguimos -∇L (gradient descent)

### 4.3 Ejemplo Visual

```python
import numpy as np
import matplotlib.pyplot as plt

# Función f(x,y) = x² + y²
def f(x, y):
    return x**2 + y**2

# Gradiente: ∇f = [2x, 2y]
def gradient(x, y):
    return np.array([2*x, 2*y])

# Crear grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Calcular gradientes
U = 2*X  # ∂f/∂x
V = 2*Y  # ∂f/∂y

plt.figure(figsize=(10, 5))

# Contour plot
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=15)
plt.colorbar()
plt.title('f(x,y) = x² + y²')

# Quiver plot (vectores gradiente)
plt.subplot(1, 2, 2)
plt.quiver(X, Y, U, V)
plt.title('Gradiente ∇f')
plt.axis('equal')
plt.show()
```

---

## 5. Chain Rule Multivariable

### 5.1 Para Redes Neuronales

En una red neuronal típica:
```
x → z = Wx + b → a = σ(z) → L (loss)
```

Queremos calcular **∂L/∂W** (cómo cambiar pesos para reducir pérdida).

**Chain rule**:
```
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
```

### 5.2 Ejemplo Concreto

**Forward pass**:
```
Capa 1: z₁ = w₁·x
Capa 2: a₁ = ReLU(z₁)
Capa 3: z₂ = w₂·a₁
Capa 4: L = (z₂ - y)²  (loss cuadrático)
```

**Backward pass** (calcular ∂L/∂w₁):
```
∂L/∂z₂ = 2(z₂ - y)
∂z₂/∂a₁ = w₂
∂a₁/∂z₁ = ReLU'(z₁) = { 1 si z₁ > 0, 0 otherwise }
∂z₁/∂w₁ = x

Chain rule:
∂L/∂w₁ = ∂L/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂w₁
       = 2(z₂-y) · w₂ · ReLU'(z₁) · x
```

**Python completo**:
```python
# Forward pass
x = 2.0
w1 = 0.5
w2 = 0.3
y = 1.0  # Target

z1 = w1 * x              # 1.0
a1 = max(0, z1)          # 1.0 (ReLU)
z2 = w2 * a1             # 0.3
L = (z2 - y)**2          # 0.49

print(f"Loss: {L}")

# Backward pass
dL_dz2 = 2 * (z2 - y)                    # -1.4
dz2_da1 = w2                              # 0.3
da1_dz1 = 1 if z1 > 0 else 0            # 1
dz1_dw1 = x                               # 2.0

dL_dw1 = dL_dz2 * dz2_da1 * da1_dz1 * dz1_dw1
print(f"∂L/∂w₁ = {dL_dw1}")  # -0.84

# Actualizar peso (gradient descent)
learning_rate = 0.1
w1_new = w1 - learning_rate * dL_dw1
print(f"w₁: {w1} → {w1_new}")  # 0.5 → 0.584
```

---

## 6. Gradient Descent

### 6.1 Idea Básica

Para **minimizar** una función f(w), iterativamente:
```
w_nuevo = w_viejo - α·∇f(w_viejo)
```

Donde:
- **α** (alpha): learning rate (tasa de aprendizaje)
- **∇f**: gradiente (dirección de subida)
- **-∇f**: dirección de bajada

### 6.2 Ejemplo: Minimizar f(w) = w²

```
f(w) = w²
f'(w) = 2w

Algoritmo:
1. Inicializar w = 5
2. Repetir:
   w = w - α·2w
```

**Python**:
```python
import numpy as np
import matplotlib.pyplot as plt

def f(w):
    return w**2

def df(w):
    return 2*w

# Inicialización
w = 5.0
alpha = 0.1
history = [w]

# Gradient descent
for i in range(20):
    w = w - alpha * df(w)
    history.append(w)
    print(f"Iteración {i+1}: w = {w:.4f}, f(w) = {f(w):.4f}")

# Visualizar
plt.plot(history, [f(w) for w in history], 'o-')
plt.xlabel('Iteración')
plt.ylabel('f(w)')
plt.title('Convergencia de Gradient Descent')
plt.grid()
plt.show()
```

### 6.3 Gradient Descent Multidimensional

Para función de múltiples variables:
```
w = [w₁, w₂, ..., wₙ]
∇f = [∂f/∂w₁, ∂f/∂w₂, ..., ∂f/∂wₙ]

Actualización:
wᵢ = wᵢ - α·∂f/∂wᵢ   para cada i
```

**Ejemplo**: Minimizar f(w₁, w₂) = w₁² + w₂²

```python
def f(w1, w2):
    return w1**2 + w2**2

def gradient(w1, w2):
    return np.array([2*w1, 2*w2])

# Inicializar
w = np.array([3.0, 4.0])
alpha = 0.1

print("Inicio:", w, "f =", f(*w))

for i in range(10):
    grad = gradient(*w)
    w = w - alpha * grad
    print(f"Iter {i+1}:", w, "f =", f(*w))

# Converge a [0, 0] (mínimo global)
```

---

## 7. Aplicaciones en Deep RL

### 7.1 Policy Gradient

Queremos maximizar retorno esperado:
```
J(θ) = E_π[Σ γᵗrₜ]
```

**Policy Gradient Theorem**:
```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) · Q^π(s,a)]
```

**Interpretación**: Aumenta probabilidad de acciones con Q alto.

**Update**:
```
θ_nuevo = θ_viejo + α·∇_θ J(θ)   (ascent, no descent!)
```

### 7.2 Q-Learning con Redes Neuronales (DQN)

**Loss**:
```
L(θ) = E[(r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]
```

**Gradiente**:
```
∇_θ L = -2·(target - Q(s,a;θ))·∇_θ Q(s,a;θ)
```

Donde target = r + γ max_a' Q(s',a';θ⁻)

### 7.3 Actor-Critic

**Actor** (política): π_θ(a|s)
**Critic** (value): V_φ(s)

**Updates**:
```
Actor:  θ = θ + α·∇_θ log π_θ(a|s)·A(s,a)
Critic: φ = φ - α·∇_φ(V_φ(s) - target)²
```

---

## 8. Herramientas: Autograd

En práctica, **NO calculas derivadas manualmente**. Frameworks como PyTorch hacen autodiferenciación:

```python
import torch

# Definir parámetros (requieren gradiente)
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])

# Forward pass
y = w * x       # y = 2*3 = 6
L = (y - 5)**2  # L = (6-5)² = 1

# Backward pass (automático!)
L.backward()

# Gradiente calculado automáticamente
print(f"∂L/∂w = {w.grad}")  # tensor([6.])

# Verificación manual: ∂L/∂w = 2(wx-5)·x = 2(6-5)·3 = 6 ✓
```

**Ejemplo más complejo**:
```python
# Red neuronal simple
x = torch.tensor([[1.0, 2.0, 3.0]])
W1 = torch.randn(3, 4, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

# Forward
h = torch.relu(x @ W1 + b1)
y_pred = h @ W2 + b2
y_true = torch.tensor([[1.0]])
loss = (y_pred - y_true)**2

# Backward
loss.backward()

# Todos los gradientes calculados!
print("∂L/∂W₁:", W1.grad.shape)  # torch.Size([3, 4])
print("∂L/∂W₂:", W2.grad.shape)  # torch.Size([4, 1])
```

---

## 9. Ejercicios Prácticos

### Ejercicio 1: Derivadas Básicas
Calcula f'(x) para:
1. f(x) = 3x² + 2x - 5
2. f(x) = x³ - 4x

<details>
<summary>Ver solución</summary>

1. f'(x) = 6x + 2
2. f'(x) = 3x² - 4
</details>

### Ejercicio 2: Chain Rule
f(x) = (2x + 1)⁴. Calcula f'(x).

<details>
<summary>Ver solución</summary>

Sea u = 2x + 1, entonces f = u⁴

f'(x) = 4u³ · 2 = 8(2x + 1)³
</details>

### Ejercicio 3: Derivadas Parciales
f(x,y) = x²y + xy². Calcula ∂f/∂x y ∂f/∂y.

<details>
<summary>Ver solución</summary>

∂f/∂x = 2xy + y² (tratar y como constante)
∂f/∂y = x² + 2xy (tratar x como constante)
</details>

### Ejercicio 4: Gradient Descent
Minimiza f(w) = (w-3)² usando gradient descent con w₀=0, α=0.1, 5 iteraciones.

<details>
<summary>Ver solución</summary>

```python
f'(w) = 2(w-3)

Iteración 1: w = 0 - 0.1·2(0-3) = 0.6
Iteración 2: w = 0.6 - 0.1·2(0.6-3) = 1.08
Iteración 3: w = 1.08 - 0.1·2(1.08-3) = 1.464
Iteración 4: w = 1.464 - 0.1·2(1.464-3) = 1.771
Iteración 5: w = 1.771 - 0.1·2(1.771-3) = 2.017
```

Converge hacia w=3 (mínimo).
</details>

---

## 10. Conceptos Avanzados (Opcional)

### 10.1 Hessian (Segunda Derivada)

La matriz Hessiana contiene segundas derivadas:
```
H = [ ∂²f/∂x₁²   ∂²f/∂x₁∂x₂ ]
    [ ∂²f/∂x₂∂x₁ ∂²f/∂x₂²   ]
```

**Uso**: Optimización de segundo orden (Newton's method), análisis de convergencia.

### 10.2 Jacobian

Para función vectorial f: ℝⁿ → ℝᵐ, el Jacobiano es matriz de derivadas:
```
J = [ ∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ ]
    [ ∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ ]
    [   ...      ...    ...    ...   ]
    [ ∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ ]
```

**En Deep RL**: Backpropagation a través de capas múltiples.

---

## 11. Cheat Sheet: Derivadas Esenciales

| Función | Derivada | Notas |
|---------|----------|-------|
| c | 0 | Constante |
| x | 1 | |
| xⁿ | nxⁿ⁻¹ | Regla del poder |
| eˣ | eˣ | Exponencial |
| ln(x) | 1/x | Logaritmo natural |
| sin(x) | cos(x) | |
| cos(x) | -sin(x) | |
| σ(x) | σ(x)(1-σ(x)) | Sigmoid |
| tanh(x) | 1-tanh²(x) | |
| ReLU(x) | {1 si x>0, 0 si x≤0} | |

**Reglas**:
- Suma: (f+g)' = f' + g'
- Producto: (fg)' = f'g + fg'
- Cadena: (f∘g)' = f'(g)·g'

---

## 12. Recursos Adicionales

### Videos (Español)
- [Khan Academy - Cálculo](https://es.khanacademy.org/math/calculus-1)
- [3Blue1Brown - Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

### Libros
- Stewart, "Calculus"
- [Deep Learning Book - Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html)

### Práctica
- [Brilliant.org - Calculus](https://brilliant.org/courses/calculus/)
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

---

## 13. Autoevaluación

¿Puedes responder?

- [ ] ¿Qué mide una derivada?
- [ ] ¿Cómo aplicar la chain rule?
- [ ] ¿Qué es un gradiente?
- [ ] ¿Cómo funciona gradient descent?
- [ ] ¿Por qué usamos -∇L en optimización?

Si respondiste todo, ¡excelente! Continúa con [Python y NumPy](04_python_numpy.md).

---

## Próximos Pasos

1. **[Python y NumPy](04_python_numpy.md)** - Programación práctica
2. **[Optimización](05_conceptos_optimizacion.md)** - Algoritmos de optimización
3. **[Fundamentos de RL](../01_fundamentos/introduccion.md)** - ¡Empezar RL!

¡Sigue adelante! 🚀
