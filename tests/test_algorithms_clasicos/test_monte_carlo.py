"""
Tests para algoritmos de Monte Carlo
=====================================

Tests unitarios para MC Prediction y MC Control (On-Policy y Off-Policy).
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/Reinforcement-learning-guide')

from algoritmos_clasicos.monte_carlo.mc_prediction import (
    MCPrediction, SimpleGridWorld, SimpleBlackjack
)
from algoritmos_clasicos.monte_carlo.mc_control import (
    MCControlOnPolicy, MCControlOffPolicy, CliffWalking
)


class TestMCPrediction:
    """Tests para Monte Carlo Prediction"""

    def test_initialization(self):
        """Test de inicialización correcta"""
        mc = MCPrediction(gamma=0.99, method='first-visit')

        assert mc.gamma == 0.99
        assert mc.method == 'first-visit'
        assert isinstance(mc.V, dict)
        assert isinstance(mc.returns, dict)

    def test_invalid_method_raises_error(self):
        """Test de que método inválido lanza error"""
        with pytest.raises(ValueError):
            MCPrediction(method='invalid-method')

    def test_generate_episode(self):
        """Test de generación de episodio"""
        env = SimpleGridWorld(size=4)
        mc = MCPrediction(gamma=0.99)

        def random_policy(state):
            return np.random.randint(0, 4)

        episode = mc.generate_episode(env, random_policy, max_steps=100)

        assert len(episode) > 0
        assert all(len(step) == 4 for step in episode)  # (s, a, r, s_next)

    def test_calculate_returns(self):
        """Test de cálculo de returns"""
        mc = MCPrediction(gamma=0.9)

        # Episodio simple: [(s, a, r, s_next), ...]
        episode = [
            ((0, 0), 0, -1, (0, 1)),
            ((0, 1), 1, -1, (0, 2)),
            ((0, 2), 1, 10, (0, 3))
        ]

        returns = mc.calculate_returns(episode)

        assert len(returns) == len(episode)
        # G_2 = 10
        assert np.isclose(returns[2], 10)
        # G_1 = -1 + 0.9*10 = 8
        assert np.isclose(returns[1], -1 + 0.9*10)
        # G_0 = -1 + 0.9*(-1 + 0.9*10) = -1 + 0.9*8 = 6.2
        assert np.isclose(returns[0], -1 + 0.9*8)

    def test_first_visit_vs_every_visit(self):
        """Test de First-Visit vs Every-Visit"""
        env = SimpleGridWorld(size=4)

        def simple_policy(state):
            return 1  # Siempre derecha

        # First-Visit
        mc_first = MCPrediction(gamma=0.99, method='first-visit')
        mc_first.evaluate_policy(env, simple_policy, num_episodes=100)

        # Every-Visit
        mc_every = MCPrediction(gamma=0.99, method='every-visit')
        mc_every.evaluate_policy(env, simple_policy, num_episodes=100)

        # Ambos deben haber aprendido valores
        assert len(mc_first.V) > 0
        assert len(mc_every.V) > 0

    def test_evaluate_policy(self):
        """Test de evaluación de política completa"""
        env = SimpleGridWorld(size=4)
        mc = MCPrediction(gamma=0.99)

        def random_policy(state):
            return np.random.randint(0, 4)

        results = mc.evaluate_policy(env, random_policy, num_episodes=500)

        assert 'V' in results
        assert 'episodes' in results
        assert 'average_return' in results
        assert results['episodes'] == 500

    def test_get_value(self):
        """Test de obtención de valor de estado"""
        mc = MCPrediction()
        mc.V[(0, 0)] = 5.5

        assert mc.get_value((0, 0)) == 5.5
        assert mc.get_value((1, 1)) == 0.0  # Estado no visitado


class TestSimpleGridWorld:
    """Tests para el ambiente SimpleGridWorld"""

    def test_initialization(self):
        """Test de inicialización del ambiente"""
        env = SimpleGridWorld(size=5)

        assert env.size == 5
        assert env.start_state == (0, 0)
        assert env.goal_state == (4, 4)
        assert env.n_actions == 4

    def test_reset(self):
        """Test de reset del ambiente"""
        env = SimpleGridWorld(size=4)
        state = env.reset()

        assert state == (0, 0)

    def test_step_valid_moves(self):
        """Test de movimientos válidos"""
        env = SimpleGridWorld(size=3)
        env.reset()

        # Mover derecha
        next_state, reward, done = env.step(1)
        assert next_state == (0, 1)
        assert not done

        # Mover abajo
        next_state, reward, done = env.step(2)
        assert next_state == (1, 1)

    def test_step_boundaries(self):
        """Test de límites del grid"""
        env = SimpleGridWorld(size=3)
        env.reset()

        # Intentar mover arriba desde (0,0)
        next_state, reward, done = env.step(0)
        assert next_state == (0, 0)  # Debe quedarse en el mismo lugar

    def test_goal_reached(self):
        """Test de alcanzar el objetivo"""
        env = SimpleGridWorld(size=2)
        env.current_state = (1, 1)  # Posición objetivo

        next_state, reward, done = env.step(0)  # Cualquier acción
        assert done


class TestMCControlOnPolicy:
    """Tests para MC Control On-Policy"""

    def test_initialization(self):
        """Test de inicialización correcta"""
        agent = MCControlOnPolicy(gamma=0.99, epsilon=0.1)

        assert agent.gamma == 0.99
        assert agent.epsilon == 0.1
        assert isinstance(agent.Q, dict)
        assert isinstance(agent.returns, dict)

    def test_get_epsilon_greedy_action(self):
        """Test de selección ε-greedy"""
        agent = MCControlOnPolicy(epsilon=0.1)

        # Establecer algunos Q-values
        state = (0, 0)
        agent.Q[state] = {0: 1.0, 1: 5.0, 2: 2.0, 3: 0.5}

        # Con epsilon=0, siempre debe elegir la mejor acción (1)
        agent.epsilon = 0.0
        actions = [agent.get_epsilon_greedy_action(state) for _ in range(10)]
        assert all(a == 1 for a in actions)

    def test_train(self):
        """Test de entrenamiento completo"""
        env = SimpleGridWorld(size=4)
        agent = MCControlOnPolicy(gamma=0.99, epsilon=0.1)

        results = agent.train(env, num_episodes=100)

        assert 'episodes' in results
        assert 'average_return' in results
        assert len(agent.Q) > 0

    def test_get_action(self):
        """Test de obtención de acción"""
        agent = MCControlOnPolicy()
        agent.Q[(0, 0)] = {0: 1.0, 1: 2.0, 2: 0.5, 3: 1.5}

        # Greedy action
        action_greedy = agent.get_action((0, 0), greedy=True)
        assert action_greedy == 1  # Mejor Q-value

        # ε-greedy puede ser exploratoria
        action_explore = agent.get_action((0, 0), greedy=False)
        assert 0 <= action_explore < 4

    def test_epsilon_decay(self):
        """Test de decaimiento de epsilon"""
        env = SimpleGridWorld(size=3)
        agent = MCControlOnPolicy(epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

        initial_epsilon = agent.epsilon
        agent.train(env, num_episodes=100)

        # Epsilon debe haber decaído
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min


class TestMCControlOffPolicy:
    """Tests para MC Control Off-Policy"""

    def test_initialization(self):
        """Test de inicialización correcta"""
        agent = MCControlOffPolicy(gamma=0.99, epsilon=0.3)

        assert agent.gamma == 0.99
        assert agent.epsilon == 0.3
        assert isinstance(agent.Q, dict)
        assert isinstance(agent.C, dict)  # Cumulative weights

    def test_get_behavior_action(self):
        """Test de selección de acción del behavior policy"""
        agent = MCControlOffPolicy(epsilon=0.2)

        state = (0, 0)
        agent.Q[state] = {0: 1.0, 1: 3.0, 2: 2.0, 3: 0.5}

        # Behavior policy es ε-greedy
        action = agent.get_behavior_action(state)
        assert 0 <= action < 4

    def test_get_target_action(self):
        """Test de selección de acción del target policy"""
        agent = MCControlOffPolicy()

        state = (0, 0)
        agent.Q[state] = {0: 1.0, 1: 3.0, 2: 2.0, 3: 0.5}

        # Target policy es greedy
        action = agent.get_target_action(state)
        assert action == 1  # Mejor Q-value

    def test_train_with_importance_sampling(self):
        """Test de entrenamiento con importance sampling"""
        env = SimpleGridWorld(size=4)
        agent = MCControlOffPolicy(gamma=0.99, epsilon=0.3)

        results = agent.train(env, num_episodes=200)

        assert 'episodes' in results
        assert 'average_return' in results
        assert 'average_importance_ratio' in results
        assert len(agent.Q) > 0

    def test_importance_ratio_calculation(self):
        """Test implícito de cálculo de importance ratio"""
        env = SimpleGridWorld(size=3)
        agent = MCControlOffPolicy(epsilon=0.5)

        # Entrenar brevemente
        agent.train(env, num_episodes=50)

        # Si se entrenó correctamente, debe tener pesos acumulados
        assert len(agent.C) > 0


class TestCliffWalking:
    """Tests para el ambiente Cliff Walking"""

    def test_initialization(self):
        """Test de inicialización del ambiente"""
        env = CliffWalking(height=4, width=12)

        assert env.height == 4
        assert env.width == 12
        assert env.start_state == (3, 0)
        assert env.goal_state == (3, 11)
        assert env.n_actions == 4

    def test_cliff_states(self):
        """Test de identificación de cliff states"""
        env = CliffWalking()

        # Estados del acantilado son (3, 1) a (3, 10)
        assert (3, 1) in env.cliff_states
        assert (3, 5) in env.cliff_states
        assert (3, 10) in env.cliff_states

        # Estos NO deben ser cliff
        assert (3, 0) not in env.cliff_states  # Start
        assert (3, 11) not in env.cliff_states  # Goal
        assert (2, 5) not in env.cliff_states

    def test_cliff_penalty(self):
        """Test de penalización por caer al acantilado"""
        env = CliffWalking()
        env.current_state = (3, 0)

        # Mover a la derecha cae al cliff
        next_state, reward, done = env.step(1)

        assert reward == -100  # Gran penalización
        assert done


class TestOnPolicyVsOffPolicy:
    """Tests comparativos entre On-Policy y Off-Policy MC Control"""

    @pytest.mark.slow
    def test_both_learn_optimal_policy(self):
        """Test de que ambos aprenden políticas óptimas"""
        env = SimpleGridWorld(size=4)

        # On-Policy
        agent_on = MCControlOnPolicy(gamma=0.99, epsilon=0.1)
        results_on = agent_on.train(env, num_episodes=1000)

        # Off-Policy
        agent_off = MCControlOffPolicy(gamma=0.99, epsilon=0.3)
        results_off = agent_off.train(env, num_episodes=1000)

        # Ambos deben aprender políticas decentes
        # (no necesariamente idénticas debido a estocasticidad)
        assert results_on['average_return'] > -10
        assert results_off['average_return'] > -10

    @pytest.mark.slow
    def test_cliff_walking_comparison(self):
        """Test de comparación en Cliff Walking"""
        env = CliffWalking()

        # On-Policy tiende a tomar camino seguro
        agent_on = MCControlOnPolicy(gamma=0.99, epsilon=0.1)
        agent_on.train(env, num_episodes=500)

        # Off-Policy puede aprender camino óptimo (arriesgado)
        agent_off = MCControlOffPolicy(gamma=0.99, epsilon=0.5)
        agent_off.train(env, num_episodes=500)

        # Ambos deben aprender algo
        assert len(agent_on.Q) > 0
        assert len(agent_off.Q) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
