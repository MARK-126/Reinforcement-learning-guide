# Notas del Tutorial Monte Carlo

Este archivo contiene la estructura y contenido del notebook interactivo de Monte Carlo.

## Estructura del Notebook

### 1. Introducción Teórica
- ¿Qué son los métodos Monte Carlo?
- Diferencias con Dynamic Programming
- Ventajas: model-free, aprendizaje por experiencia
- Desventajas: alta varianza, requiere episodios completos

### 2. Fundamentos Matemáticos

**Monte Carlo Prediction:**
- Estimación de V^π(s) mediante promedio de returns
- First-Visit: V(s) = average(G_t | s visitado por primera vez en episodio)
- Every-Visit: V(s) = average(G_t | s visitado en cualquier momento)

**Return (Retorno):**
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t}R_T

### 3. MC Prediction - Ejemplos Completos
- GridWorld 5x5
- Blackjack con estrategias fijas
- Comparación First-Visit vs Every-Visit

### 4. MC Control On-Policy
- ε-greedy policy improvement
- Algoritmo GLIE (Greedy in the Limit with Infinite Exploration)
- Convergencia a política óptima

### 5. MC Control Off-Policy
- Importance Sampling
- Weighted importance sampling para reducir varianza
- Separación entre behavior policy y target policy

### 6. Experimentos
- Blackjack: diferentes estrategias
- Cliff Walking: on-policy vs off-policy
- Análisis de convergencia
- Efecto de epsilon en exploración

### 7. Visualizaciones
- Curvas de aprendizaje
- Heatmaps de políticas
- Distribución de returns
- Trayectorias óptimas

### 8. Ejercicios Prácticos
1. Implementar política custom para Blackjack
2. Analizar convergencia con diferentes valores de ε
3. Comparar First-Visit vs Every-Visit empíricamente
4. Crear un ambiente episódico nuevo
5. Implementar MC Control con decaimiento de ε

## Ejecución

Para usar el notebook completo, ejecutar en orden:
1. Imports y configuración
2. Teoría y ejemplos visuales
3. Experimentos prácticos
4. Ejercicios hands-on

El notebook está diseñado para 3-4 horas de estudio activo.
