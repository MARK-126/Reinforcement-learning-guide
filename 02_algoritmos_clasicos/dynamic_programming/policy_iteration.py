"""
Policy Iteration para Reinforcement Learning
============================================

Implementación del algoritmo de Policy Iteration (Iteración de Política)
para resolver Markov Decision Processes (MDPs).

Policy Iteration alterna entre dos pasos:
1. Policy Evaluation: Calcular V^π para la política actual
2. Policy Improvement: Mejorar la política usando el greedy policy con respecto a V^π

Autor: MARK-126
"""

import numpy as np
from typing import Dict, Tuple, List, Callable
from collections import defaultdict


class PolicyIteration:
    """
    Algoritmo de Policy Iteration para MDPs discretos.

    Policy Iteration encuentra la política óptima mediante iteración alternada
    entre evaluación de política y mejora de política.

    Parámetros:
    -----------
    n_states : int
        Número de estados en el MDP
    n_actions : int
        Número de acciones posibles
    gamma : float (default=0.99)
        Factor de descuento (discount factor)
    theta : float (default=1e-6)
        Umbral de convergencia para policy evaluation

    Atributos:
    ----------
    V : np.ndarray
        Función de valor estado (state-value function)
    policy : np.ndarray
        Política actual (determinista)
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

        # Inicializar política aleatoria (uniforme)
        self.policy = np.random.randint(0, n_actions, size=n_states)

        # Historial de entrenamiento
        self.history = {
            'iterations': 0,
            'policy_changes': [],
            'max_value_changes': []
        }

    def policy_evaluation(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray
    ) -> np.ndarray:
        """
        Policy Evaluation: Calcula V^π para la política actual.

        Usa iterative policy evaluation (Sutton & Barto, Sección 4.1):
        V(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]

        Parámetros:
        -----------
        transition_probs : np.ndarray, shape (n_states, n_actions, n_states)
            Probabilidades de transición p(s'|s,a)
        rewards : np.ndarray, shape (n_states, n_actions, n_states)
            Recompensas r(s,a,s')

        Retorna:
        --------
        V : np.ndarray
            Función de valor para la política actual
        """
        while True:
            delta = 0
            V_old = self.V.copy()

            for s in range(self.n_states):
                # Acción seleccionada por la política actual
                a = self.policy[s]

                # Calcular valor esperado: Σ_{s'} p(s'|s,a)[r(s,a,s') + γV(s')]
                v = 0
                for s_next in range(self.n_states):
                    prob = transition_probs[s, a, s_next]
                    reward = rewards[s, a, s_next]
                    v += prob * (reward + self.gamma * V_old[s_next])

                self.V[s] = v
                delta = max(delta, abs(V_old[s] - self.V[s]))

            # Convergencia: si el cambio máximo es menor que theta
            if delta < self.theta:
                break

        return self.V

    def policy_improvement(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Policy Improvement: Mejora la política usando greedy selection.

        Para cada estado s, selecciona la acción que maximiza:
        Q(s,a) = Σ_{s'} p(s'|s,a)[r(s,a,s') + γV(s')]

        Parámetros:
        -----------
        transition_probs : np.ndarray
            Probabilidades de transición
        rewards : np.ndarray
            Recompensas

        Retorna:
        --------
        new_policy : np.ndarray
            Nueva política mejorada
        policy_stable : bool
            True si la política no cambió (convergencia)
        """
        policy_stable = True
        new_policy = self.policy.copy()

        for s in range(self.n_states):
            # Acción anterior
            old_action = self.policy[s]

            # Calcular Q(s,a) para todas las acciones
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    prob = transition_probs[s, a, s_next]
                    reward = rewards[s, a, s_next]
                    q_values[a] += prob * (reward + self.gamma * self.V[s_next])

            # Seleccionar la mejor acción (greedy)
            best_action = np.argmax(q_values)
            new_policy[s] = best_action

            # Verificar si la política cambió
            if old_action != best_action:
                policy_stable = False

        return new_policy, policy_stable

    def solve(
        self,
        transition_probs: np.ndarray,
        rewards: np.ndarray,
        max_iterations: int = 1000
    ) -> Dict:
        """
        Ejecuta Policy Iteration hasta convergencia.

        Parámetros:
        -----------
        transition_probs : np.ndarray, shape (n_states, n_actions, n_states)
            Matriz de transición del MDP
        rewards : np.ndarray, shape (n_states, n_actions, n_states)
            Matriz de recompensas
        max_iterations : int
            Número máximo de iteraciones

        Retorna:
        --------
        results : dict
            Diccionario con política óptima, valores y estadísticas
        """
        print("Iniciando Policy Iteration...")

        for iteration in range(max_iterations):
            # 1. Policy Evaluation
            self.policy_evaluation(transition_probs, rewards)

            # 2. Policy Improvement
            new_policy, policy_stable = self.policy_improvement(
                transition_probs, rewards
            )

            # Registrar cambios
            policy_changes = np.sum(new_policy != self.policy)
            self.history['policy_changes'].append(policy_changes)

            self.policy = new_policy

            print(f"Iteración {iteration + 1}: {policy_changes} estados cambiaron su política")

            # Convergencia: política estable
            if policy_stable:
                print(f"\n✓ Convergencia alcanzada en {iteration + 1} iteraciones")
                self.history['iterations'] = iteration + 1
                break
        else:
            print(f"\n⚠ Alcanzado máximo de iteraciones ({max_iterations})")
            self.history['iterations'] = max_iterations

        return {
            'policy': self.policy,
            'V': self.V,
            'iterations': self.history['iterations'],
            'history': self.history
        }

    def get_action(self, state: int) -> int:
        """
        Selecciona acción según la política aprendida.

        Parámetros:
        -----------
        state : int
            Estado actual

        Retorna:
        --------
        action : int
            Acción a tomar
        """
        return self.policy[state]

    def get_value(self, state: int) -> float:
        """
        Retorna el valor de un estado.

        Parámetros:
        -----------
        state : int
            Estado

        Retorna:
        --------
        value : float
            V(s) bajo la política actual
        """
        return self.V[state]


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
    Ejemplo de uso: Resolver GridWorld 4x4 con Policy Iteration
    """
    print("=" * 60)
    print("Policy Iteration - Ejemplo: GridWorld 4x4")
    print("=" * 60)

    # Crear MDP
    transition_probs, rewards, n_states, n_actions = create_gridworld_mdp(
        grid_size=4,
        goal_reward=1.0,
        step_reward=-0.01
    )

    # Crear solver
    solver = PolicyIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=0.99,
        theta=1e-6
    )

    # Resolver
    results = solver.solve(transition_probs, rewards)

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

    print(f"\nFunción de valor:")
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            print(f"{results['V'][state]:6.3f} ", end="")
        print()

    print(f"\nIteraciones totales: {results['iterations']}")
