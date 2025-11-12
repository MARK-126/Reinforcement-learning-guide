# CONTRIBUTING.md

¡Gracias por tu interés en contribuir a esta guía de Reinforcement Learning! 🎉

## Cómo Contribuir

### Reportar Errores

Si encuentras un error en el código, documentación, o explicaciones:

1. Verifica que el error no haya sido reportado antes en [Issues](https://github.com/MARK-126/Reinforcement-learning-guide/issues)
2. Crea un nuevo issue con:
   - Descripción clara del error
   - Pasos para reproducirlo (si aplica)
   - Comportamiento esperado vs actual
   - Capturas de pantalla si es relevante

### Sugerir Mejoras

¿Tienes ideas para mejorar el contenido?

1. Abre un issue describiendo tu sugerencia
2. Explica por qué sería útil
3. Si es posible, proporciona ejemplos o referencias

### Contribuir con Código

#### 1. Fork y Clone

```bash
# Fork el repositorio en GitHub, luego:
git clone https://github.com/TU_USUARIO/Reinforcement-learning-guide.git
cd Reinforcement-learning-guide
```

#### 2. Crea una Rama

```bash
git checkout -b feature/mi-contribucion
```

#### 3. Realiza tus Cambios

- **Código**: Sigue el estilo existente (PEP 8 para Python)
- **Documentación**: Escribe en español, claro y conciso
- **Comentarios**: Explica el "por qué", no solo el "qué"

#### 4. Prueba tus Cambios

```bash
# Asegúrate de que el código funciona
python tu_archivo.py

# Si añadiste dependencias, actualiza requirements.txt
pip freeze > requirements.txt
```

#### 5. Commit y Push

```bash
git add .
git commit -m "Descripción clara de tus cambios"
git push origin feature/mi-contribucion
```

#### 6. Abre un Pull Request

1. Ve a tu fork en GitHub
2. Click en "Compare & pull request"
3. Describe tus cambios detalladamente
4. Espera revisión y feedback

## Guías de Estilo

### Python

- **PEP 8**: Sigue las convenciones de Python
- **Docstrings**: Usa docstrings para funciones y clases
- **Type hints**: Cuando sea posible, añade type hints

```python
def train_agent(env: gym.Env, episodes: int = 1000) -> List[float]:
    """
    Entrena un agente en el ambiente dado.
    
    Args:
        env: Ambiente de Gymnasium
        episodes: Número de episodios de entrenamiento
    
    Returns:
        Lista de recompensas por episodio
    """
    pass
```

### Markdown

- **Encabezados**: Usa jerarquía clara (# > ## > ###)
- **Código**: Especifica el lenguaje en bloques de código
- **Enlaces**: Usa enlaces descriptivos
- **Listas**: Consistente (- o 1. 2. 3.)

### Estructura de Archivos

```
/algoritmo/
├── README.md           # Explicación del algoritmo
├── algorithm.py        # Implementación
└── example.py          # Ejemplo de uso
```

## Tipos de Contribuciones Bienvenidas

### 📚 Documentación
- Mejorar explicaciones existentes
- Añadir ejemplos
- Traducir contenido
- Corregir errores tipográficos

### 💻 Código
- Implementar algoritmos faltantes
- Optimizar código existente
- Añadir tests
- Mejorar visualizaciones

### 🎓 Contenido Educativo
- Tutoriales nuevos
- Ejercicios prácticos
- Notebooks de Jupyter
- Diagramas y visualizaciones

### 🐛 Correcciones
- Bugs en código
- Errores en matemáticas
- Enlaces rotos
- Formato inconsistente

## Contenido que Buscamos

### Algoritmos
- Métodos tabulares (Monte Carlo, TD, etc.)
- Deep RL (Rainbow DQN, A2C, etc.)
- Policy Gradient avanzados
- Model-based RL
- Multi-agent RL
- Meta-RL

### Ejemplos
- Ambientes clásicos (CartPole, LunarLander, etc.)
- Custom environments
- Aplicaciones reales
- Visualizaciones interactivas

### Recursos
- Papers importantes
- Implementaciones de referencia
- Datasets
- Benchmarks

## Lo que NO Aceptamos

- Código copiado sin atribución
- Contenido plagiado
- Implementaciones sin documentación
- Código que no funciona
- Contenido ofensivo o inapropiado

## Proceso de Revisión

1. **Revisión inicial** (1-3 días): Verificamos que el PR cumple requisitos básicos
2. **Revisión técnica** (3-7 días): Revisamos código y contenido
3. **Feedback**: Te daremos feedback constructivo
4. **Iteración**: Harás cambios basados en feedback
5. **Merge**: Una vez aprobado, hacemos merge

## Reconocimientos

Todos los contribuidores serán reconocidos en:
- README.md (sección de contribuidores)
- Historial de commits de Git
- Release notes

## Preguntas

¿Tienes preguntas? 
- Abre un [issue de discusión](https://github.com/MARK-126/Reinforcement-learning-guide/issues)
- Etiquétalo como "question"

## Código de Conducta

### Nuestro Compromiso

Crear un ambiente acogedor y respetuoso para todos.

### Comportamientos Esperados

- Ser respetuoso con diferentes puntos de vista
- Aceptar críticas constructivas
- Enfocarse en lo mejor para la comunidad
- Mostrar empatía hacia otros miembros

### Comportamientos Inaceptables

- Lenguaje ofensivo o inapropiado
- Trolling o comentarios insultantes
- Acoso público o privado
- Publicar información privada de otros sin permiso

## Licencia

Al contribuir, aceptas que tus contribuciones serán licenciadas bajo la MIT License del proyecto.

---

¡Gracias por contribuir a hacer de esta la mejor guía de RL en español! 🚀🤖
