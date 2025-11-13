"""
Value Iteration para Reinforcement Learning
===========================================

Implementación del algoritmo de Value Iteration (Iteración de Valor)
para resolver Markov Decision Processes (MDPs).

Value Iteration combina policy evaluation y policy improvement en un solo paso:
V(s) ← max_a Σ_{s'} p(s'|s,a)[r(s,a,s') + γV(s')]

Es más eficiente que Policy Iteration para muchos problemas.

Autor: MARK-126
"""

import numpy as np
from typing import Dict, Tuple
from collections import defaultdict


class ValueIteration:
    """
    Algoritmo de Value Iteration para MDPs discretos.

    Value Iteration encuentra la función de valor óptima mediante
    actualizaciones iterativas usando la ecuación de optimalidad de Bellman.

    Parámetros:
    -----------
    n_states : int
        Número de estados en el MDP
    n_actions : int
        Número de acciones posibles
    gamma : float (default=0.99)
        Factor de descuento (discount factor)
    theta : float (default=1e-6)
        Umbral de convergencia

    Atributos:
    ----------
    V : np.ndarray
        Función de valor óptima V*
    policy : np.ndarray
        Política óptima derivada de V*
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        theta: float = 1e-6
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta

        # Inicializar V(s) = 0 para todos los estados
        self.V = np.zeros(n_states)

        # Política será extraída al final
        self.policy = None

        # Historial de entrenamiento
        self.history = {
            'iterations': 0,
            'max_deltas': [],
            'mean_values': []
        }

    def value_update(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray
    ) -> float:
        """
        Realiza una iteración de value update.

        Ecuación de Bellman Optimality:
        V(s) ← max_a Σ_{s'} p(s'|s,a)[r(s,a,s') + γV(s')]

        Parámetros:
        -----------
        transition_probs : np.ndarray, shape (n_states, n_actions, n_states)
            Probabilidades de transición p(s'|s,a)
        rewards : np.ndarray, shape (n_states, n_actions, n_states)
            Recompensas r(s,a,s')

        Retorna:
        --------
        delta : float
            Cambio máximo en V durante esta iteración
        """
        delta = 0
        V_old = self.V.copy()

        for s in range(self.n_states):
            # Calcular Q(s,a) para todas las acciones
            q_values = np.zeros(self.n_actions)

            for a in range(self.n_actions):
                # Q(s,a) = Σ_{s'} p(s'|s,a)[r(s,a,s') + γV(s')]
                for s_next in range(self.n_states):
                    prob = transition_probs[s, a, s_next]
                    reward = rewards[s, a, s_next]
                    q_values[a] += prob * (reward + self.gamma * V_old[s_next])

            # V(s) ← max_a Q(s,a)
            self.V[s] = np.max(q_values)

            # Registrar cambio
            delta = max(delta, abs(V_old[s] - self.V[s]))

        return delta

    def extract_policy(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray
    ) -> np.ndarray:
        """
        Extrae la política óptima de la función de valor óptima.

        Para cada estado, selecciona la acción que maximiza:
        π*(s) = argmax_a Σ_{s'} p(s'|s,a)[r(s,a,s') + γV*(s')]

        Parámetros:
        -----------
        transition_probs : np.ndarray
            Probabilidades de transición
        rewards : np.ndarray
            Recompensas

        Retorna:
        --------
        policy : np.ndarray
            Política óptima
        """
        policy = np.zeros(self.n_states, dtype=int)

        for s in range(self.n_states):
            # Calcular Q(s,a) para todas las acciones
            q_values = np.zeros(self.n_actions)

            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    prob = transition_probs[s, a, s_next]
                    reward = rewards[s, a, s_next]
                    q_values[a] += prob * (reward + self.gamma * self.V[s_next])

            # Seleccionar mejor acción
            policy[s] = np.argmax(q_values)

        return policy

    def solve(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray,
        max_iterations: int = 1000,
        verbose: bool = True
    ) -> Dict:
        """
        Ejecuta Value Iteration hasta convergencia.

        Parámetros:
        -----------
        transition_probs : np.ndarray, shape (n_states, n_actions, n_states)
            Matriz de transición del MDP
        rewards : np.ndarray, shape (n_states, n_actions, n_states)
            Matriz de recompensas
        max_iterations : int
            Número máximo de iteraciones
        verbose : bool
            Si True, imprime progreso

        Retorna:
        --------
        results : dict
            Diccionario con política óptima, valores y estadísticas
        """
        if verbose:
            print("Iniciando Value Iteration...")

        for iteration in range(max_iterations):
            # Actualizar valores
            delta = self.value_update(transition_probs, rewards)

            # Registrar estadísticas
            self.history['max_deltas'].append(delta)
            self.history['mean_values'].append(np.mean(self.V))

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteración {iteration + 1}: delta = {delta:.6f}, "
                      f"V_mean = {np.mean(self.V):.4f}")

            # Convergencia
            if delta < self.theta:
                if verbose:
                    print(f"\n✓ Convergencia alcanzada en {iteration + 1} iteraciones")
                self.history['iterations'] = iteration + 1
                break
        else:
            if verbose:
                print(f"\n⚠ Alcanzado máximo de iteraciones ({max_iterations})")
            self.history['iterations'] = max_iterations

        # Extraer política óptima
        self.policy = self.extract_policy(transition_probs, rewards)

        return {
            'policy': self.policy,
            'V': self.V,
            'iterations': self.history['iterations'],
            'history': self.history
        }

    def get_action(self, state: int) -> int:
        """
        Selecciona acción según la política óptima.

        Parámetros:
        -----------
        state : int
            Estado actual

        Retorna:
        --------
        action : int
            Acción óptima
        """
        if self.policy is None:
            raise ValueError("Debe ejecutar solve() primero para obtener la política")
        return self.policy[state]

    def get_value(self, state: int) -> float:
        """
        Retorna el valor óptimo de un estado.

        Parámetros:
        -----------
        state : int
            Estado

        Retorna:
        --------
        value : float
            V*(s)
        """
        return self.V[state]

    def get_q_value(
        self,
        state: int,
        action: int,
        transition_probs: np.ndarray,
        rewards: np.ndarray
    ) -> float:
        """
        Calcula Q*(s,a) para un par estado-acción.

        Parámetros:
        -----------
        state : int
            Estado
        action : int
            Acción
        transition_probs : np.ndarray
            Probabilidades de transición
        rewards : np.ndarray
            Recompensas

        Retorna:
        --------
        q_value : float
            Q*(s,a)
        """
        q_value = 0
        for s_next in range(self.n_states):
            prob = transition_probs[state, action, s_next]
            reward = rewards[state, action, s_next]
            q_value += prob * (reward + self.gamma * self.V[s_next])
        return q_value


def create_gridworld_mdp(
    grid_size: int = 4,
    goal_reward: float = 1.0,
    step_reward: float = -0.01
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Crea un GridWorld simple para demostración.

    El agente se mueve en una cuadrícula de grid_size x grid_size.
    Objetivo: llegar a la esquina inferior derecha.

    Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda

    Parámetros:
    -----------
    grid_size : int
        Tamaño de la cuadrícula
    goal_reward : float
        Recompensa al llegar al objetivo
    step_reward : float
        Recompensa (costo) por cada paso

    Retorna:
    --------
    transition_probs : np.ndarray
        Probabilidades de transición
    rewards : np.ndarray
        Recompensas
    n_states : int
        Número de estados
    n_actions : int
        Número de acciones
    """
    n_states = grid_size * grid_size
    n_actions = 4  # arriba, derecha, abajo, izquierda

    # Estado objetivo (esquina inferior derecha)
    goal_state = n_states - 1

    # Inicializar matrices
    transition_probs = np.zeros((n_states, n_actions, n_states))
    rewards = np.zeros((n_states, n_actions, n_states))

    def state_to_pos(state):
        """Convierte estado a posición (row, col)"""
        return state // grid_size, state % grid_size

    def pos_to_state(row, col):
        """Convierte posición a estado"""
        return row * grid_size + col

    # Definir transiciones
    for s in range(n_states):
        if s == goal_state:
            # Estado terminal: permanece en el objetivo
            for a in range(n_actions):
                transition_probs[s, a, s] = 1.0
                rewards[s, a, s] = 0.0
            continue

        row, col = state_to_pos(s)

        # Para cada acción
        for a in range(n_actions):
            # Calcular nueva posición
            new_row, new_col = row, col

            if a == 0:  # arriba
                new_row = max(0, row - 1)
            elif a == 1:  # derecha
                new_col = min(grid_size - 1, col + 1)
            elif a == 2:  # abajo
                new_row = min(grid_size - 1, row + 1)
            elif a == 3:  # izquierda
                new_col = max(0, col - 1)

            next_state = pos_to_state(new_row, new_col)

            # Transición determinista
            transition_probs[s, a, next_state] = 1.0

            # Recompensa
            if next_state == goal_state:
                rewards[s, a, next_state] = goal_reward
            else:
                rewards[s, a, next_state] = step_reward

    return transition_probs, rewards, n_states, n_actions


if __name__ == "__main__":
    """
    Ejemplo de uso: Resolver GridWorld 4x4 con Value Iteration
    """
    print("=" * 60)
    print("Value Iteration - Ejemplo: GridWorld 4x4")
    print("=" * 60)

    # Crear MDP
    transition_probs, rewards, n_states, n_actions = create_gridworld_mdp(
        grid_size=4,
        goal_reward=1.0,
        step_reward=-0.01
    )

    # Crear solver
    solver = ValueIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=0.99,
        theta=1e-6
    )

    # Resolver
    results = solver.solve(transition_probs, rewards, verbose=True)

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    print(f"\nPolítica óptima (0=↑, 1=→, 2=↓, 3=←):")
    policy_symbols = ['↑', '→', '↓', '←']
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            action = results['policy'][state]
            print(f" {policy_symbols[action]} ", end="")
        print()

    print(f"\nFunción de valor óptima:")
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            print(f"{results['V'][state]:6.3f} ", end="")
        print()

    print(f"\nIteraciones totales: {results['iterations']}")

    # Comparar eficiencia
    print(f"\nConvergencia:")
    print(f"- Delta final: {results['history']['max_deltas'][-1]:.8f}")
    print(f"- Valor medio: {np.mean(results['V']):.4f}")
