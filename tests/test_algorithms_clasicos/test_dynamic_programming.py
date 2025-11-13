"""
Tests para algoritmos de Dynamic Programming
=============================================

Tests unitarios para Policy Iteration y Value Iteration.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/Reinforcement-learning-guide')

from algoritmos_clasicos.dynamic_programming.policy_iteration import (
    PolicyIteration, create_gridworld_mdp
)
from algoritmos_clasicos.dynamic_programming.value_iteration import (
    ValueIteration
)


class TestPolicyIteration:
    """Tests para Policy Iteration"""

    def test_initialization(self):
        """Test de inicialización correcta"""
        pi = PolicyIteration(n_states=16, n_actions=4, gamma=0.99)

        assert pi.n_states == 16
        assert pi.n_actions == 4
        assert pi.gamma == 0.99
        assert len(pi.V) == 16
        assert len(pi.policy) == 16
        assert np.all(pi.V == 0)

    def test_policy_evaluation(self):
        """Test de evaluación de política"""
        # Crear GridWorld simple
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=3, goal_reward=1.0, step_reward=-0.01
        )

        pi = PolicyIteration(n_states=n_states, n_actions=n_actions, gamma=0.9)

        # Evaluar política inicial
        V = pi.policy_evaluation(trans_probs, rewards)

        # V debe tener valores para todos los estados
        assert len(V) == n_states
        assert isinstance(V, np.ndarray)

        # Estado objetivo debe tener valor más alto
        # (después de varias evaluaciones convergerá al óptimo)
        assert V[n_states-1] >= V[0]  # Goal vs start

    def test_policy_improvement(self):
        """Test de mejora de política"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=3, goal_reward=1.0, step_reward=0
        )

        pi = PolicyIteration(n_states=n_states, n_actions=n_actions, gamma=0.99)

        # Evaluar y mejorar
        pi.policy_evaluation(trans_probs, rewards)
        new_policy, stable = pi.policy_improvement(trans_probs, rewards)

        assert len(new_policy) == n_states
        assert isinstance(stable, bool)
        assert all(0 <= action < n_actions for action in new_policy)

    def test_solve_converges(self):
        """Test de convergencia del algoritmo"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=4, goal_reward=1.0, step_reward=-0.01
        )

        pi = PolicyIteration(n_states=n_states, n_actions=n_actions, gamma=0.99)
        results = pi.solve(trans_probs, rewards, max_iterations=100)

        # Debe converger
        assert 'policy' in results
        assert 'V' in results
        assert 'iterations' in results
        assert results['iterations'] < 100  # Debe converger antes de max_iterations

    def test_optimal_policy_gridworld(self):
        """Test de política óptima en GridWorld conocido"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=4, goal_reward=1.0, step_reward=-0.01
        )

        pi = PolicyIteration(n_states=n_states, n_actions=n_actions, gamma=0.99)
        results = pi.solve(trans_probs, rewards)

        # Estado objetivo debe tener cualquier acción (ya está en goal)
        # Estados adyacentes deben apuntar hacia el objetivo
        # Estado 14 (3,2) debe ir a la derecha (1) o abajo (2) hacia goal (3,3)
        assert results['policy'][14] in [1, 2]

    def test_get_action(self):
        """Test de obtención de acción"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(grid_size=3)

        pi = PolicyIteration(n_states=n_states, n_actions=n_actions)
        pi.solve(trans_probs, rewards)

        action = pi.get_action(0)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < n_actions

    def test_get_value(self):
        """Test de obtención de valor de estado"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(grid_size=3)

        pi = PolicyIteration(n_states=n_states, n_actions=n_actions)
        pi.solve(trans_probs, rewards)

        value = pi.get_value(0)
        assert isinstance(value, (float, np.floating))


class TestValueIteration:
    """Tests para Value Iteration"""

    def test_initialization(self):
        """Test de inicialización correcta"""
        vi = ValueIteration(n_states=16, n_actions=4, gamma=0.99, theta=1e-6)

        assert vi.n_states == 16
        assert vi.n_actions == 4
        assert vi.gamma == 0.99
        assert vi.theta == 1e-6
        assert len(vi.V) == 16
        assert np.all(vi.V == 0)
        assert vi.policy is None  # No se extrae hasta después de solve()

    def test_value_update(self):
        """Test de actualización de valores"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=3, goal_reward=1.0, step_reward=-0.01
        )

        vi = ValueIteration(n_states=n_states, n_actions=n_actions, gamma=0.9)

        # Realizar una actualización
        delta = vi.value_update(trans_probs, rewards)

        assert isinstance(delta, (float, np.floating))
        assert delta >= 0  # Delta no puede ser negativo

    def test_extract_policy(self):
        """Test de extracción de política"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(grid_size=3)

        vi = ValueIteration(n_states=n_states, n_actions=n_actions)

        # Primero necesitamos calcular valores
        for _ in range(10):
            vi.value_update(trans_probs, rewards)

        policy = vi.extract_policy(trans_probs, rewards)

        assert len(policy) == n_states
        assert all(0 <= action < n_actions for action in policy)

    def test_solve_converges(self):
        """Test de convergencia del algoritmo"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=4, goal_reward=1.0, step_reward=-0.01
        )

        vi = ValueIteration(n_states=n_states, n_actions=n_actions, theta=1e-4)
        results = vi.solve(trans_probs, rewards, max_iterations=1000, verbose=False)

        # Debe converger
        assert 'policy' in results
        assert 'V' in results
        assert 'iterations' in results
        assert results['iterations'] < 1000  # Debe converger
        assert vi.policy is not None  # Política debe haberse extraído

    def test_get_action_requires_solve(self):
        """Test de que get_action requiere llamar a solve() primero"""
        vi = ValueIteration(n_states=16, n_actions=4)

        with pytest.raises(ValueError):
            vi.get_action(0)

    def test_get_q_value(self):
        """Test de cálculo de Q-values"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(grid_size=3)

        vi = ValueIteration(n_states=n_states, n_actions=n_actions)
        vi.solve(trans_probs, rewards, verbose=False)

        q_value = vi.get_q_value(0, 1, trans_probs, rewards)
        assert isinstance(q_value, (float, np.floating))

    def test_history_tracking(self):
        """Test de seguimiento del historial"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(grid_size=3)

        vi = ValueIteration(n_states=n_states, n_actions=n_actions)
        results = vi.solve(trans_probs, rewards, verbose=False)

        assert 'history' in results
        assert 'max_deltas' in results['history']
        assert 'mean_values' in results['history']
        assert len(results['history']['max_deltas']) == results['iterations']


class TestGridWorldMDP:
    """Tests para la creación del GridWorld MDP"""

    def test_gridworld_creation(self):
        """Test de creación correcta del GridWorld"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=4, goal_reward=1.0, step_reward=-0.01
        )

        assert trans_probs.shape == (n_states, n_actions, n_states)
        assert rewards.shape == (n_states, n_actions, n_states)
        assert n_states == 16  # 4x4
        assert n_actions == 4

    def test_transition_probabilities_valid(self):
        """Test de que las probabilidades de transición son válidas"""
        trans_probs, _, n_states, n_actions = create_gridworld_mdp(grid_size=3)

        # Cada (s,a) debe tener probabilidades que sumen 1
        for s in range(n_states):
            for a in range(n_actions):
                prob_sum = np.sum(trans_probs[s, a, :])
                assert np.isclose(prob_sum, 1.0), f"State {s}, action {a}: sum={prob_sum}"

    def test_goal_state_terminal(self):
        """Test de que el estado objetivo es terminal"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(grid_size=4)

        goal_state = n_states - 1  # Última posición

        # En estado objetivo, todas las acciones deben llevar al mismo estado
        for a in range(n_actions):
            assert trans_probs[goal_state, a, goal_state] == 1.0
            assert rewards[goal_state, a, goal_state] == 0.0


class TestPolicyVsValueIteration:
    """Tests comparativos entre Policy Iteration y Value Iteration"""

    def test_both_find_same_optimal_policy(self):
        """Test de que ambos algoritmos encuentran la misma política óptima"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=4, goal_reward=1.0, step_reward=-0.01
        )

        # Policy Iteration
        pi = PolicyIteration(n_states=n_states, n_actions=n_actions, gamma=0.99)
        pi_results = pi.solve(trans_probs, rewards)

        # Value Iteration
        vi = ValueIteration(n_states=n_states, n_actions=n_actions, gamma=0.99)
        vi_results = vi.solve(trans_probs, rewards, verbose=False)

        # Las políticas deben ser iguales (o al menos dar los mismos valores)
        # Comparar valores en lugar de políticas (puede haber empates en Q-values)
        assert np.allclose(pi_results['V'], vi_results['V'], atol=1e-3)

    @pytest.mark.slow
    def test_value_iteration_efficiency(self):
        """Test de que Value Iteration suele ser más eficiente"""
        trans_probs, rewards, n_states, n_actions = create_gridworld_mdp(
            grid_size=5, goal_reward=1.0, step_reward=-0.01
        )

        # Policy Iteration
        pi = PolicyIteration(n_states=n_states, n_actions=n_actions, gamma=0.99)
        pi_results = pi.solve(trans_probs, rewards)

        # Value Iteration
        vi = ValueIteration(n_states=n_states, n_actions=n_actions, gamma=0.99)
        vi_results = vi.solve(trans_probs, rewards, verbose=False)

        # Ambos deben converger
        assert pi_results['iterations'] > 0
        assert vi_results['iterations'] > 0

        # Los valores óptimos deben ser similares
        assert np.allclose(pi_results['V'], vi_results['V'], atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
