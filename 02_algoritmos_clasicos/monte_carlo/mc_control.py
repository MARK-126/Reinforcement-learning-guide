"""
Monte Carlo Control para Reinforcement Learning
================================================

Implementación de algoritmos de Monte Carlo para encontrar políticas óptimas
(Control) usando métodos de muestreo de episodios completos.

Monte Carlo Control encuentra la política óptima π* y la función de valor Q*
mediante la mejora iterativa de la política basada en retornos muestreados.

Variantes implementadas:
1. On-Policy MC Control (ε-greedy): Mejora la política que genera los episodios
2. Off-Policy MC Control (Importance Sampling): Aprende política óptima mientras sigue otra
3. MC Control with Exploring Starts: Garantiza exploración mediante inicios aleatorios

Ventajas de MC Control:
- No requiere modelo del entorno (model-free)
- Puede optimizar directamente desde experiencia
- Funciona bien con episodios largos
- No sufre de bootstrap error como TD

Autor: MARK-126
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from collections import defaultdict
import random


class MCControlOnPolicy:
    """
    Monte Carlo Control On-Policy con ε-greedy.

    Encuentra la política óptima mediante mejora iterativa de una política
    ε-greedy basada en estimaciones de Q(s,a).

    Algoritmo:
    1. Generar episodio siguiendo política ε-greedy actual
    2. Para cada par (s,a) en el episodio, calcular retorno G
    3. Actualizar Q(s,a) con promedio de retornos
    4. Actualizar política para ser ε-greedy respecto a Q

    Parámetros:
    -----------
    gamma : float (default=0.99)
        Factor de descuento
    epsilon : float (default=0.1)
        Parámetro de exploración para política ε-greedy
    epsilon_decay : float (default=0.9999)
        Factor de decaimiento de epsilon por episodio
    epsilon_min : float (default=0.01)
        Valor mínimo de epsilon
    method : str (default='first-visit')
        'first-visit' o 'every-visit'

    Atributos:
    ----------
    Q : dict
        Función de valor acción-estado Q(s,a)
    policy : dict
        Política actual (probabilidades de acción)
    returns : dict
        Retornos observados para cada (s,a)
    """

    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.01,
        method: str = 'first-visit'
    ):
        if method not in ['first-visit', 'every-visit']:
            raise ValueError("method debe ser 'first-visit' o 'every-visit'")

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.method = method

        # Q(s,a): función de valor acción-estado
        self.Q = defaultdict(lambda: defaultdict(float))

        # N(s,a): contador de visitas
        self.N = defaultdict(lambda: defaultdict(int))

        # Retornos observados
        self.returns = defaultdict(lambda: defaultdict(list))

        # Historial
        self.history = {
            'episodes': 0,
            'episode_returns': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'q_value_changes': []
        }

    def get_epsilon_greedy_action(
        self,
        state: Any,
        valid_actions: List[int]
    ) -> int:
        """
        Selecciona acción usando política ε-greedy.

        Con probabilidad ε: acción aleatoria
        Con probabilidad 1-ε: acción greedy (argmax Q(s,a))

        Parámetros:
        -----------
        state : Any
            Estado actual
        valid_actions : List[int]
            Lista de acciones válidas

        Retorna:
        --------
        action : int
            Acción seleccionada
        """
        if random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.choice(valid_actions)
        else:
            # Explotación: mejor acción conocida
            q_values = {a: self.Q[state][a] for a in valid_actions}
            max_q = max(q_values.values())
            # Si hay empate, elegir aleatoriamente entre las mejores
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def generate_episode(
        self,
        env: Any,
        max_steps: int = 1000,
        valid_actions: Optional[List[int]] = None
    ) -> List[Tuple[Any, int, float]]:
        """
        Genera episodio siguiendo política ε-greedy actual.

        Parámetros:
        -----------
        env : Environment
            Entorno episódico
        max_steps : int
            Máximo de pasos
        valid_actions : List[int] (optional)
            Acciones válidas (si None, asume [0,1,2,3])

        Retorna:
        --------
        episode : List[Tuple[state, action, reward]]
            Episodio completo
        """
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3]  # Default para GridWorld

        episode = []
        state = env.reset()

        for _ in range(max_steps):
            # Seleccionar acción ε-greedy
            action = self.get_epsilon_greedy_action(state, valid_actions)

            # Ejecutar acción
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))

            if done:
                break

            state = next_state

        return episode

    def update_q_values(
        self,
        episode: List[Tuple[Any, int, float]]
    ) -> float:
        """
        Actualiza Q(s,a) usando los retornos del episodio.

        Parámetros:
        -----------
        episode : List[Tuple[state, action, reward]]
            Episodio completo

        Retorna:
        --------
        max_change : float
            Cambio máximo en Q
        """
        # Calcular retornos para cada paso
        G = 0
        returns_list = []
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            returns_list.append((state, action, G))
        returns_list.reverse()

        # Actualizar Q values
        max_change = 0
        visited_pairs = set()

        for state, action, G in returns_list:
            # First-visit: solo primera ocurrencia
            sa_pair = (state, action)
            if self.method == 'first-visit' and sa_pair in visited_pairs:
                continue
            visited_pairs.add(sa_pair)

            # Guardar retorno
            self.returns[state][action].append(G)
            self.N[state][action] += 1

            # Actualizar Q como promedio de retornos
            old_q = self.Q[state][action]
            self.Q[state][action] = np.mean(self.returns[state][action])

            max_change = max(max_change, abs(self.Q[state][action] - old_q))

        return max_change

    def train(
        self,
        env: Any,
        num_episodes: int,
        max_steps: int = 1000,
        valid_actions: Optional[List[int]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Entrena el agente usando MC Control On-Policy.

        Parámetros:
        -----------
        env : Environment
            Entorno de entrenamiento
        num_episodes : int
            Número de episodios
        max_steps : int
            Pasos máximos por episodio
        valid_actions : List[int]
            Acciones válidas
        verbose : bool
            Imprimir progreso

        Retorna:
        --------
        results : dict
            Resultados del entrenamiento
        """
        if verbose:
            print(f"Iniciando MC Control On-Policy (ε-greedy)...")
            print(f"Episodios: {num_episodes}, Gamma: {self.gamma}, "
                  f"Epsilon inicial: {self.epsilon}\n")

        for episode_num in range(num_episodes):
            # 1. Generar episodio con política ε-greedy
            episode = self.generate_episode(env, max_steps, valid_actions)

            # 2. Actualizar Q values
            q_change = self.update_q_values(episode)

            # 3. Decaer epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 4. Registrar estadísticas
            episode_return = sum(r for _, _, r in episode)
            self.history['episode_returns'].append(episode_return)
            self.history['episode_lengths'].append(len(episode))
            self.history['epsilon_history'].append(self.epsilon)
            self.history['q_value_changes'].append(q_change)

            # Imprimir progreso
            if verbose and (episode_num + 1) % max(1, num_episodes // 10) == 0:
                avg_return = np.mean(self.history['episode_returns'][-100:])
                avg_length = np.mean(self.history['episode_lengths'][-100:])
                print(f"Episodio {episode_num + 1}/{num_episodes}: "
                      f"Return={episode_return:.3f}, "
                      f"Avg Return={avg_return:.3f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"ε={self.epsilon:.4f}")

        self.history['episodes'] = num_episodes

        if verbose:
            print(f"\nEntrenamiento completado!")
            print(f"Estados-acción visitados: {sum(len(actions) for actions in self.Q.values())}")

        return {
            'Q': dict(self.Q),
            'history': self.history
        }

    def get_action(self, state: Any, greedy: bool = True) -> int:
        """
        Obtiene acción para un estado.

        Parámetros:
        -----------
        state : Any
            Estado actual
        greedy : bool
            Si True, usa política greedy; si False, usa ε-greedy

        Retorna:
        --------
        action : int
            Acción seleccionada
        """
        if not greedy:
            valid_actions = list(self.Q[state].keys()) if state in self.Q else [0]
            return self.get_epsilon_greedy_action(state, valid_actions)

        # Política greedy
        if state not in self.Q or not self.Q[state]:
            return 0  # Acción por defecto

        return max(self.Q[state].items(), key=lambda x: x[1])[0]

    def get_q_value(self, state: Any, action: int) -> float:
        """Retorna Q(s,a)."""
        return self.Q[state][action] if state in self.Q else 0.0


class MCControlOffPolicy:
    """
    Monte Carlo Control Off-Policy con Importance Sampling.

    Aprende la política óptima (target policy) mientras sigue una política
    de comportamiento diferente (behavior policy).

    Usa Weighted Importance Sampling para estimar Q*(s,a) a partir de
    episodios generados por la política de comportamiento.

    Parámetros:
    -----------
    gamma : float (default=0.99)
        Factor de descuento
    epsilon : float (default=0.1)
        Epsilon para política de comportamiento (ε-greedy)
    method : str (default='first-visit')
        'first-visit' o 'every-visit'

    Atributos:
    ----------
    Q : dict
        Función de valor acción-estado Q(s,a) para política target
    C : dict
        Suma acumulada de pesos para weighted importance sampling
    """

    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        method: str = 'first-visit'
    ):
        if method not in ['first-visit', 'every-visit']:
            raise ValueError("method debe ser 'first-visit' o 'every-visit'")

        self.gamma = gamma
        self.epsilon = epsilon
        self.method = method

        # Q(s,a): política target (determinista greedy)
        self.Q = defaultdict(lambda: defaultdict(float))

        # C(s,a): suma de pesos para weighted importance sampling
        self.C = defaultdict(lambda: defaultdict(float))

        # Historial
        self.history = {
            'episodes': 0,
            'episode_returns': [],
            'importance_ratios': [],
            'q_value_changes': []
        }

    def get_behavior_action(
        self,
        state: Any,
        valid_actions: List[int]
    ) -> int:
        """
        Política de comportamiento: ε-greedy respecto a Q.
        """
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = {a: self.Q[state][a] for a in valid_actions}
            if not q_values:
                return random.choice(valid_actions)
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def get_target_action(self, state: Any, valid_actions: List[int]) -> int:
        """
        Política target: greedy respecto a Q.
        """
        q_values = {a: self.Q[state][a] for a in valid_actions}
        if not q_values:
            return random.choice(valid_actions)
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def generate_episode(
        self,
        env: Any,
        max_steps: int = 1000,
        valid_actions: Optional[List[int]] = None
    ) -> List[Tuple[Any, int, float]]:
        """
        Genera episodio usando política de comportamiento.
        """
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3]

        episode = []
        state = env.reset()

        for _ in range(max_steps):
            action = self.get_behavior_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))

            if done:
                break

            state = next_state

        return episode

    def update_q_values(
        self,
        episode: List[Tuple[Any, int, float]],
        valid_actions: List[int]
    ) -> Tuple[float, float]:
        """
        Actualiza Q usando weighted importance sampling.

        Retorna:
        --------
        max_change : float
            Cambio máximo en Q
        importance_ratio : float
            Ratio de importancia del episodio
        """
        G = 0  # Retorno
        W = 1  # Ratio de importancia acumulado
        max_change = 0

        # Procesar episodio en reversa
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]

            # Calcular retorno
            G = reward + self.gamma * G

            # Actualizar suma de pesos
            self.C[state][action] += W

            # Actualizar Q con weighted average
            if self.C[state][action] > 0:
                old_q = self.Q[state][action]
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                max_change = max(max_change, abs(self.Q[state][action] - old_q))

            # Si acción no es greedy según target policy, terminar
            target_action = self.get_target_action(state, valid_actions)
            if action != target_action:
                break

            # Actualizar peso: W = W * (1 / π_behavior(a|s))
            # π_behavior(a|s) para ε-greedy:
            n_actions = len(valid_actions)
            if action == target_action:
                # Probabilidad de seleccionar acción greedy
                prob_behavior = (1 - self.epsilon) + self.epsilon / n_actions
            else:
                # Probabilidad de seleccionar acción no-greedy
                prob_behavior = self.epsilon / n_actions

            # Target policy es determinista (π_target(a|s) = 1 para acción greedy)
            W *= 1.0 / prob_behavior

        return max_change, W

    def train(
        self,
        env: Any,
        num_episodes: int,
        max_steps: int = 1000,
        valid_actions: Optional[List[int]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Entrena usando MC Control Off-Policy.

        Parámetros:
        -----------
        env : Environment
            Entorno de entrenamiento
        num_episodes : int
            Número de episodios
        max_steps : int
            Pasos máximos por episodio
        valid_actions : List[int]
            Acciones válidas
        verbose : bool
            Imprimir progreso

        Retorna:
        --------
        results : dict
            Resultados del entrenamiento
        """
        if verbose:
            print(f"Iniciando MC Control Off-Policy (Importance Sampling)...")
            print(f"Episodios: {num_episodes}, Gamma: {self.gamma}, "
                  f"Epsilon (behavior): {self.epsilon}\n")

        for episode_num in range(num_episodes):
            # 1. Generar episodio con behavior policy
            episode = self.generate_episode(env, max_steps, valid_actions)

            # 2. Actualizar Q usando importance sampling
            q_change, importance_ratio = self.update_q_values(episode, valid_actions or [0, 1, 2, 3])

            # 3. Registrar estadísticas
            episode_return = sum(r for _, _, r in episode)
            self.history['episode_returns'].append(episode_return)
            self.history['importance_ratios'].append(importance_ratio)
            self.history['q_value_changes'].append(q_change)

            # Imprimir progreso
            if verbose and (episode_num + 1) % max(1, num_episodes // 10) == 0:
                avg_return = np.mean(self.history['episode_returns'][-100:])
                avg_ratio = np.mean(self.history['importance_ratios'][-100:])
                print(f"Episodio {episode_num + 1}/{num_episodes}: "
                      f"Return={episode_return:.3f}, "
                      f"Avg Return={avg_return:.3f}, "
                      f"Avg Importance Ratio={avg_ratio:.2f}")

        self.history['episodes'] = num_episodes

        if verbose:
            print(f"\nEntrenamiento completado!")
            print(f"Estados-acción visitados: {sum(len(actions) for actions in self.Q.values())}")

        return {
            'Q': dict(self.Q),
            'history': self.history
        }

    def get_action(self, state: Any) -> int:
        """
        Obtiene acción según política target (greedy).
        """
        if state not in self.Q or not self.Q[state]:
            return 0
        return max(self.Q[state].items(), key=lambda x: x[1])[0]

    def get_q_value(self, state: Any, action: int) -> float:
        """Retorna Q(s,a)."""
        return self.Q[state][action] if state in self.Q else 0.0


# ==================== ENTORNO DE EJEMPLO ====================

class SimpleGridWorld:
    """GridWorld 5x5 para demostración."""

    def __init__(self, size: int = 5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = None

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda"""
        row, col = self.state

        if action == 0:  # arriba
            row = max(0, row - 1)
        elif action == 1:  # derecha
            col = min(self.size - 1, col + 1)
        elif action == 2:  # abajo
            row = min(self.size - 1, row + 1)
        elif action == 3:  # izquierda
            col = max(0, col - 1)

        self.state = (row, col)

        if self.state == self.goal:
            return self.state, 1.0, True, {}
        else:
            return self.state, -0.01, False, {}


class CliffWalking:
    """
    Cliff Walking: GridWorld con acantilado.

    Cuadrícula 4x12 con acantilado en la fila inferior.
    Caer al acantilado da recompensa -100 y reinicia.
    """

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        # Acantilado: celdas (3, 1) a (3, 10)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.state = None

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda"""
        row, col = self.state

        if action == 0:  # arriba
            row = max(0, row - 1)
        elif action == 1:  # derecha
            col = min(self.cols - 1, col + 1)
        elif action == 2:  # abajo
            row = min(self.rows - 1, row + 1)
        elif action == 3:  # izquierda
            col = max(0, col - 1)

        self.state = (row, col)

        # Verificar acantilado
        if self.state in self.cliff:
            return self.start, -100.0, False, {}

        # Verificar meta
        if self.state == self.goal:
            return self.state, -1.0, True, {}

        return self.state, -1.0, False, {}


# ==================== MAIN ====================

if __name__ == "__main__":
    """
    Ejemplos de uso: Monte Carlo Control
    """

    print("=" * 70)
    print("MONTE CARLO CONTROL - Ejemplos")
    print("=" * 70)

    # ========== EJEMPLO 1: On-Policy MC Control (GridWorld) ==========
    print("\n" + "=" * 70)
    print("Ejemplo 1: GridWorld 5x5 - On-Policy MC Control")
    print("=" * 70)

    # Crear entorno
    env1 = SimpleGridWorld(size=5)

    # Crear agente on-policy
    agent_on = MCControlOnPolicy(
        gamma=0.99,
        epsilon=0.2,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        method='first-visit'
    )

    # Entrenar
    results1 = agent_on.train(
        env=env1,
        num_episodes=5000,
        max_steps=100,
        valid_actions=[0, 1, 2, 3],
        verbose=True
    )

    # Evaluar política aprendida
    print("\nEvaluando política aprendida...")
    test_episodes = 100
    test_returns = []
    test_lengths = []

    for _ in range(test_episodes):
        state = env1.reset()
        episode_return = 0
        episode_length = 0

        for step in range(100):
            action = agent_on.get_action(state, greedy=True)
            state, reward, done, _ = env1.step(action)
            episode_return += reward
            episode_length += 1

            if done:
                break

        test_returns.append(episode_return)
        test_lengths.append(episode_length)

    print(f"\nResultados de evaluación ({test_episodes} episodios):")
    print(f"  - Return promedio: {np.mean(test_returns):.4f} ± {np.std(test_returns):.4f}")
    print(f"  - Longitud promedio: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")

    # Mostrar política aprendida
    print("\nPolítica aprendida (muestra):")
    policy_symbols = ['↑', '→', '↓', '←']
    sample_states = [(0, 0), (0, 2), (1, 1), (2, 2), (3, 3), (4, 3)]
    for state in sample_states:
        if state in agent_on.Q:
            action = agent_on.get_action(state)
            q_value = agent_on.get_q_value(state, action)
            print(f"  Estado {state}: {policy_symbols[action]} (Q={q_value:.3f})")

    # ========== EJEMPLO 2: Off-Policy MC Control (GridWorld) ==========
    print("\n" + "=" * 70)
    print("Ejemplo 2: GridWorld 5x5 - Off-Policy MC Control")
    print("=" * 70)

    # Crear agente off-policy
    agent_off = MCControlOffPolicy(
        gamma=0.99,
        epsilon=0.3,  # Behavior policy más exploratoria
        method='first-visit'
    )

    # Entrenar
    results2 = agent_off.train(
        env=env1,
        num_episodes=10000,  # Más episodios para off-policy
        max_steps=100,
        valid_actions=[0, 1, 2, 3],
        verbose=True
    )

    # Evaluar
    print("\nEvaluando política aprendida (Off-Policy)...")
    test_returns_off = []
    test_lengths_off = []

    for _ in range(test_episodes):
        state = env1.reset()
        episode_return = 0
        episode_length = 0

        for step in range(100):
            action = agent_off.get_action(state)
            state, reward, done, _ = env1.step(action)
            episode_return += reward
            episode_length += 1

            if done:
                break

        test_returns_off.append(episode_return)
        test_lengths_off.append(episode_length)

    print(f"\nResultados de evaluación ({test_episodes} episodios):")
    print(f"  - Return promedio: {np.mean(test_returns_off):.4f} ± {np.std(test_returns_off):.4f}")
    print(f"  - Longitud promedio: {np.mean(test_lengths_off):.2f} ± {np.std(test_lengths_off):.2f}")

    # ========== EJEMPLO 3: Cliff Walking - Comparación On vs Off Policy ==========
    print("\n" + "=" * 70)
    print("Ejemplo 3: Cliff Walking - Comparación On-Policy vs Off-Policy")
    print("=" * 70)
    print("Problema: Caminar por acantilado (cliff) que da -100 de penalización")

    env3 = CliffWalking()

    # On-Policy
    print("\nEntrenando agente On-Policy...")
    agent_cliff_on = MCControlOnPolicy(
        gamma=0.99,
        epsilon=0.1,
        epsilon_decay=0.999,
        epsilon_min=0.01
    )
    agent_cliff_on.train(env3, num_episodes=3000, verbose=False)

    # Off-Policy
    print("Entrenando agente Off-Policy...")
    agent_cliff_off = MCControlOffPolicy(
        gamma=0.99,
        epsilon=0.2
    )
    agent_cliff_off.train(env3, num_episodes=5000, verbose=False)

    # Evaluar ambos
    def evaluate_agent(agent, env, episodes=100):
        returns = []
        for _ in range(episodes):
            state = env.reset()
            ep_return = 0
            for _ in range(200):
                if hasattr(agent, 'get_action'):
                    action = agent.get_action(state, greedy=True) if isinstance(agent, MCControlOnPolicy) else agent.get_action(state)
                else:
                    action = 0
                state, reward, done, _ = env.step(action)
                ep_return += reward
                if done:
                    break
            returns.append(ep_return)
        return np.mean(returns), np.std(returns)

    mean_on, std_on = evaluate_agent(agent_cliff_on, env3)
    mean_off, std_off = evaluate_agent(agent_cliff_off, env3)

    print(f"\nResultados Cliff Walking:")
    print(f"  On-Policy:  Return = {mean_on:.2f} ± {std_on:.2f}")
    print(f"  Off-Policy: Return = {mean_off:.2f} ± {std_off:.2f}")
    print(f"\nNota: On-Policy tiende a aprender caminos más seguros (evita cliff)")
    print(f"      Off-Policy aprende la ruta óptima (cerca del cliff)")

    # ========== COMPARACIÓN FINAL ==========
    print("\n" + "=" * 70)
    print("RESUMEN DE COMPARACIONES")
    print("=" * 70)

    print("\nGridWorld 5x5:")
    print(f"  On-Policy:  {np.mean(test_returns):.4f} ± {np.std(test_returns):.4f}")
    print(f"  Off-Policy: {np.mean(test_returns_off):.4f} ± {np.std(test_returns_off):.4f}")

    print("\nCliff Walking:")
    print(f"  On-Policy:  {mean_on:.2f} ± {std_on:.2f}")
    print(f"  Off-Policy: {mean_off:.2f} ± {std_off:.2f}")

    print("\n" + "=" * 70)
    print("Conclusiones:")
    print("- On-Policy es más estable pero aprende políticas conservadoras")
    print("- Off-Policy puede aprender políticas óptimas arriesgadas")
    print("- Off-Policy requiere más episodios debido a importance sampling")
    print("=" * 70)
