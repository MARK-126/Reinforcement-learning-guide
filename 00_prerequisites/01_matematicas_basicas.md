# Matemáticas Básicas para Reinforcement Learning

## 🎯 Objetivo de Esta Sección

Esta guía está diseñada para personas **sin conocimientos matemáticos previos** más allá de aritmética básica. Te prepararemos paso a paso para entender los conceptos matemáticos que necesitarás en Reinforcement Learning.

---

## 1. Notación Matemática Básica

Antes de empezar, familiarícémonos con símbolos que verás frecuentemente:

| Símbolo | Significado | Ejemplo |
|---------|-------------|---------|
| **∈** | "pertenece a" | x ∈ S significa "x está en el conjunto S" |
| **Σ** | "suma de" | Σᵢ xᵢ = x₁ + x₂ + x₃ + ... |
| **∏** | "producto de" | ∏ᵢ xᵢ = x₁ × x₂ × x₃ × ... |
| **≈** | "aproximadamente igual" | 3.14159 ≈ 3.14 |
| **≥, ≤** | "mayor o igual", "menor o igual" | x ≥ 5 |
| **\|** | "tal que" | {x \| x > 0} = "conjunto de x tal que x es mayor que 0" |

### Ejemplo Práctico: Sumatoria (Σ)

**Pregunta**: ¿Cuánto es Σᵢ₌₁³ i²?

**Solución paso a paso**:
```
Σᵢ₌₁³ i² = 1² + 2² + 3²
         = 1 + 4 + 9
         = 14
```

**En Python**:
```python
suma = sum([i**2 for i in range(1, 4)])  # resultado: 14
```

---

## 2. Probabilidad desde Cero

### 2.1 ¿Qué es Probabilidad?

La **probabilidad** mide qué tan probable es que algo suceda. Va de 0 (imposible) a 1 (seguro).

**Ejemplos cotidianos**:
- Lanzar una moneda: P(cara) = 0.5 (50%)
- Tirar un dado: P(sacar 6) = 1/6 ≈ 0.167 (16.7%)
- Lluvia mañana: P(lluvia) = 0.3 (30%)

### 2.2 Cálculo de Probabilidad Básica

```
P(evento) = número de casos favorables / número de casos totales
```

**Ejemplo**: Tienes una bolsa con 3 bolas rojas y 7 bolas azules.

```
P(roja) = 3/(3+7) = 3/10 = 0.3
P(azul) = 7/10 = 0.7
```

**En código**:
```python
import random

# Simulación de 1000 extracciones
bolsa = ['roja']*3 + ['azul']*7
extracciones = [random.choice(bolsa) for _ in range(1000)]
prob_roja = extracciones.count('roja') / 1000
print(f"P(roja) ≈ {prob_roja}")  # Aproximadamente 0.3
```

### 2.3 Probabilidad Condicional

**Pregunta**: ¿Qué probabilidad hay de que llueva Y haga frío?

La **probabilidad condicional** P(A|B) se lee "probabilidad de A dado que B ocurrió".

```
P(A|B) = P(A y B) / P(B)
```

**Ejemplo real**:
- P(enfermo | test positivo) = probabilidad de estar enfermo dado que el test salió positivo

**Ejemplo con números**:
```
P(lluvia) = 0.3
P(frío | lluvia) = 0.8  (80% de los días lluviosos hace frío)
P(lluvia Y frío) = P(lluvia) × P(frío | lluvia)
                 = 0.3 × 0.8 = 0.24
```

### 2.4 Eventos Independientes

Dos eventos son **independientes** si uno no afecta al otro.

**Ejemplos**:
- Lanzar dos monedas: el resultado de la primera no afecta la segunda
- P(cara₁ Y cara₂) = P(cara₁) × P(cara₂) = 0.5 × 0.5 = 0.25

**En RL**: Las transiciones en un MDP son probabilísticas pero independientes del pasado (propiedad de Markov).

---

## 3. Estadística Esencial

### 3.1 Media (Promedio)

La **media** es el valor promedio de un conjunto de números.

```
media = (x₁ + x₂ + ... + xₙ) / n = (Σᵢ xᵢ) / n
```

**Ejemplo**:
Recompensas en 5 episodios: [10, 15, 12, 18, 20]

```
media = (10 + 15 + 12 + 18 + 20) / 5 = 75 / 5 = 15
```

**En Python**:
```python
import numpy as np

recompensas = [10, 15, 12, 18, 20]
media = np.mean(recompensas)  # 15.0
```

### 3.2 Varianza y Desviación Estándar

La **varianza** mide qué tan dispersos están los datos. La **desviación estándar** es su raíz cuadrada (más interpretable).

```
varianza = Σᵢ (xᵢ - media)² / n
desviación_estándar = √varianza
```

**Ejemplo**:
Recompensas: [10, 15, 12, 18, 20], media = 15

```
varianza = [(10-15)² + (15-15)² + (12-15)² + (18-15)² + (20-15)²] / 5
         = [25 + 0 + 9 + 9 + 25] / 5
         = 68 / 5 = 13.6

desviación_estándar = √13.6 ≈ 3.69
```

**Interpretación**: Las recompensas típicamente varían ± 3.69 alrededor del promedio (15).

**En Python**:
```python
varianza = np.var(recompensas)  # 13.6
std = np.std(recompensas)       # 3.69
```

### 3.3 Valor Esperado (Expected Value)

El **valor esperado** E[X] es el promedio que esperarías obtener si repitieras un experimento aleatorio muchas veces.

```
E[X] = Σᵢ xᵢ · P(xᵢ)
```

**Ejemplo**: Lanzas un dado. ¿Cuál es el valor esperado?

```
E[dado] = 1·(1/6) + 2·(1/6) + 3·(1/6) + 4·(1/6) + 5·(1/6) + 6·(1/6)
        = (1 + 2 + 3 + 4 + 5 + 6) / 6
        = 21 / 6 = 3.5
```

**En RL**: El valor de un estado V(s) es el retorno esperado (recompensa promedio futura).

**Simulación en Python**:
```python
import random

# Simular 10000 lanzamientos de dado
lanzamientos = [random.randint(1, 6) for _ in range(10000)]
promedio = sum(lanzamientos) / len(lanzamientos)
print(f"E[dado] ≈ {promedio}")  # Aproximadamente 3.5
```

---

## 4. Distribuciones de Probabilidad

### 4.1 Distribución Uniforme

Todos los resultados tienen la **misma probabilidad**.

**Ejemplo**: Dado justo
```
P(1) = P(2) = P(3) = P(4) = P(5) = P(6) = 1/6
```

**En Python**:
```python
import matplotlib.pyplot as plt

resultados = [random.randint(1, 6) for _ in range(1000)]
plt.hist(resultados, bins=6, density=True)
plt.title("Distribución Uniforme (Dado)")
plt.show()
```

### 4.2 Distribución Normal (Gaussiana)

La famosa "curva de campana". La mayoría de valores están cerca de la media.

**Parámetros**:
- **μ (mu)**: media (centro de la campana)
- **σ (sigma)**: desviación estándar (ancho de la campana)

**Ejemplo**: Alturas humanas siguen una distribución normal
```
μ = 170 cm
σ = 10 cm
```

**En Python**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Generar 1000 valores de una normal
valores = np.random.normal(loc=170, scale=10, size=1000)
plt.hist(valores, bins=30, density=True)
plt.title("Distribución Normal (μ=170, σ=10)")
plt.show()
```

### 4.3 Distribución de Bernoulli

Experimento con solo **dos resultados**: éxito (1) o fracaso (0).

**Ejemplo**: Lanzar una moneda
```
P(éxito) = p = 0.5
P(fracaso) = 1-p = 0.5
```

**En RL**: Ambientes estocásticos pueden usar Bernoulli para determinar transiciones.

**En Python**:
```python
# Lanzar moneda 10 veces
lanzamientos = np.random.binomial(n=1, p=0.5, size=10)
print(lanzamientos)  # Ej: [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
```

---

## 5. Operaciones con Funciones

### 5.1 Máximo y Mínimo

**max**: Encuentra el valor más grande
**min**: Encuentra el valor más pequeño

```python
valores = [3, 7, 2, 9, 1]
maximo = max(valores)  # 9
minimo = min(valores)  # 1
```

### 5.2 argmax y argmin

**argmax**: Encuentra el **índice** (posición) del valor máximo
**argmin**: Encuentra el **índice** del valor mínimo

**Ejemplo**:
```python
import numpy as np

Q = [0.2, 0.8, 0.5, 0.9, 0.3]
#    ↑    ↑    ↑    ↑    ↑
#    0    1    2    3    4  (índices)

mejor_accion = np.argmax(Q)  # 3 (posición del 0.9)
```

**En RL**: `argmax Q(s,a)` significa "selecciona la acción con mayor valor Q".

### 5.3 Función Exponencial

La función **exp(x) = eˣ** donde e ≈ 2.718 es la base del logaritmo natural.

**Propiedades importantes**:
```
exp(0) = 1
exp(1) ≈ 2.718
exp(-∞) → 0
exp(∞) → ∞
```

**En Python**:
```python
import numpy as np

np.exp(0)   # 1.0
np.exp(1)   # 2.718...
np.exp(-5)  # 0.0067... (muy pequeño)
```

**En RL**: Función softmax usa exponenciales para convertir valores en probabilidades.

---

## 6. Conceptos Avanzados (Preparación para RL)

### 6.1 Descuento Geométrico (γ - gamma)

En RL, recompensas futuras valen **menos** que recompensas presentes. Esto se modela con **descuento geométrico**.

```
Retorno total = r₁ + γr₂ + γ²r₃ + γ³r₄ + ...
```

Donde **γ** (gamma) ∈ [0, 1] es el factor de descuento.

**Ejemplo**:
- Recompensas: r₁=10, r₂=10, r₃=10
- γ = 0.9

```
Retorno = 10 + 0.9·10 + 0.9²·10
        = 10 + 9 + 8.1
        = 27.1
```

**Interpretación**:
- γ = 0: Solo importa recompensa inmediata (agente miope)
- γ = 1: Todas las recompensas valen igual (agente previsor infinito)
- γ = 0.9: Balance típico en RL

**En Python**:
```python
def calcular_retorno(recompensas, gamma):
    """Calcula retorno descontado"""
    retorno = 0
    for t, r in enumerate(recompensas):
        retorno += (gamma ** t) * r
    return retorno

recompensas = [10, 10, 10, 10, 10]
retorno = calcular_retorno(recompensas, gamma=0.9)
print(f"Retorno con γ=0.9: {retorno:.2f}")  # 40.95
```

### 6.2 Series Geométricas Infinitas

Si γ < 1, la suma infinita tiene un valor finito:

```
Σₜ₌₀^∞ γᵗ = 1 + γ + γ² + γ³ + ... = 1/(1-γ)
```

**Ejemplo** (γ = 0.9):
```
1/(1-0.9) = 1/0.1 = 10
```

**En RL**: Si todas las recompensas son 1, el retorno máximo es 1/(1-γ).

### 6.3 Convergencia y Límites

Una secuencia **converge** si se acerca cada vez más a un valor límite.

**Ejemplo**:
```
Secuencia: 1, 0.5, 0.25, 0.125, 0.0625, ...
Límite: 0
```

**En RL**: Algoritmos iterativos convergen cuando valores dejan de cambiar significativamente.

```python
def ha_convergido(valor_anterior, valor_nuevo, threshold=1e-6):
    """Verifica si un valor ha convergido"""
    return abs(valor_nuevo - valor_anterior) < threshold

# Ejemplo
V_anterior = 10.5
V_nuevo = 10.5000001
if ha_convergido(V_anterior, V_nuevo):
    print("¡Convergió!")
```

---

## 7. Ejercicios Prácticos

### Ejercicio 1: Probabilidad Básica
Tienes una baraja de 52 cartas. ¿Cuál es la probabilidad de sacar un As?

<details>
<summary>Ver solución</summary>

```
P(As) = 4/52 = 1/13 ≈ 0.077 (7.7%)
```

Hay 4 ases en 52 cartas totales.
</details>

### Ejercicio 2: Valor Esperado
Un juego te da +$10 con probabilidad 0.6 y -$5 con probabilidad 0.4. ¿Cuál es la ganancia esperada?

<details>
<summary>Ver solución</summary>

```
E[ganancia] = 10·0.6 + (-5)·0.4
            = 6 - 2
            = $4
```

En promedio, ganas $4 por juego.
</details>

### Ejercicio 3: Descuento
Recompensas: [5, 5, 5], γ = 0.8. Calcula el retorno total.

<details>
<summary>Ver solución</summary>

```
G = 5 + 0.8·5 + 0.8²·5
  = 5 + 4 + 3.2
  = 12.2
```
</details>

### Ejercicio 4: argmax
Q_values = [0.3, 0.7, 0.5, 0.9, 0.2]. ¿Cuál es argmax Q?

<details>
<summary>Ver solución</summary>

```
argmax Q = 3
```

El valor máximo es 0.9, que está en el índice 3.
</details>

---

## 8. Cheat Sheet: Fórmulas Esenciales

| Concepto | Fórmula | Uso en RL |
|----------|---------|-----------|
| **Probabilidad** | P(A) = casos_favorables / casos_totales | Transiciones estocásticas |
| **Esperanza** | E[X] = Σ xᵢ·P(xᵢ) | Valor de estados V(s) |
| **Media** | μ = Σxᵢ / n | Promedio de recompensas |
| **Varianza** | σ² = Σ(xᵢ-μ)² / n | Estabilidad del aprendizaje |
| **Descuento** | G = Σ γᵗrₜ | Retorno total |
| **Serie geométrica** | Σγᵗ = 1/(1-γ) | Horizonte infinito |

---

## 9. Recursos Adicionales

### Videos Recomendados (Español)
- [Khan Academy - Probabilidad](https://es.khanacademy.org/math/probability)
- [Khan Academy - Estadística](https://es.khanacademy.org/math/statistics-probability)

### Práctica Interactiva
- [Brilliant.org - Probability](https://brilliant.org/courses/probability/)
- [Coursera - Data Science Math Skills](https://www.coursera.org/learn/datasciencemathskills)

### Para Python
- [Python for Data Analysis](https://wesmckinney.com/book/)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

---

## 10. Autoevaluación

¿Estás listo para continuar? Deberías poder responder:

- [ ] ¿Qué significa P(A|B)?
- [ ] ¿Cómo se calcula un valor esperado?
- [ ] ¿Qué hace argmax?
- [ ] ¿Por qué usamos γ (descuento) en RL?
- [ ] ¿Cuándo converge una serie geométrica?

Si respondiste todo, ¡estás listo para [Álgebra Lineal](02_algebra_lineal.md)!

---

## Próximos Pasos

1. **[Álgebra Lineal](02_algebra_lineal.md)** - Vectores y matrices
2. **[Cálculo Básico](03_calculo_basico.md)** - Derivadas y gradientes
3. **[Python y NumPy](04_python_numpy.md)** - Programación para RL
4. **[Optimización](05_conceptos_optimizacion.md)** - Encontrar mejores soluciones

¡Sigue adelante! 🚀
