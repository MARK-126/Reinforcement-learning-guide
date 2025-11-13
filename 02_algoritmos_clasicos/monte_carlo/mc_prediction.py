"""
Monte Carlo Prediction para Reinforcement Learning
===================================================

Implementación de algoritmos de Monte Carlo para evaluación de políticas
(Policy Evaluation) usando métodos de muestreo de episodios completos.

Monte Carlo Prediction estima la función de valor V^π para una política dada
mediante el promedio de retornos observados en múltiples episodios.

Variantes implementadas:
- First-Visit MC: Promedia solo la primera visita a cada estado en un episodio
- Every-Visit MC: Promedia todas las visitas a cada estado

Ventajas de Monte Carlo:
- No requiere modelo del entorno (model-free)
- Puede aprender de experiencia real o simulada
- Simple de implementar y entender
- Funciona bien con dominios de estados grandes

Autor: MARK-126
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from collections import defaultdict
import random


class MCPrediction:
    """
    Algoritmo de Monte Carlo Prediction para evaluación de políticas.

    Estima la función de valor V^π(s) para una política dada usando
    el promedio de retornos observados en episodios muestreados.

    Parámetros:
    -----------
    gamma : float (default=0.99)
        Factor de descuento (discount factor)
    method : str (default='first-visit')
        Método de MC: 'first-visit' o 'every-visit'

    Atributos:
    ----------
    V : dict
        Función de valor estado (state-value function) V^π(s)
    returns : dict
        Lista de retornos observados para cada estado
    visit_counts : dict
        Contador de visitas a cada estado
    """

    def __init__(
        self,
        gamma: float = 0.99,
        method: str = 'first-visit'
    ):
        if method not in ['first-visit', 'every-visit']:
            raise ValueError("method debe ser 'first-visit' o 'every-visit'")

        self.gamma = gamma
        self.method = method

        # Función de valor: V(s)
        self.V = defaultdict(float)

        # Retornos observados para cada estado
        self.returns = defaultdict(list)

        # Contadores de visitas
        self.visit_counts = defaultdict(int)

        # Historial de entrenamiento
        self.history = {
            'episodes': 0,
            'mean_returns': [],
            'value_changes': [],
            'states_visited': []
        }

    def generate_episode(
        self,
        env: Any,
        policy: Callable[[Any], int],
        max_steps: int = 1000
    ) -> List[Tuple[Any, int, float]]:
        """
        Genera un episodio completo siguiendo la política dada.

        Un episodio es una secuencia de (estado, acción, recompensa) desde
        un estado inicial hasta un estado terminal.

        Parámetros:
        -----------
        env : Environment
            Entorno episódico (debe tener reset() y step())
        policy : Callable
            Política a seguir: policy(state) -> action
        max_steps : int
            Número máximo de pasos por episodio

        Retorna:
        --------
        episode : List[Tuple[state, action, reward]]
            Lista de tuplas (s, a, r) del episodio
        """
        episode = []
        state = env.reset()

        for _ in range(max_steps):
            # Seleccionar acción según la política
            action = policy(state)

            # Ejecutar acción en el entorno
            next_state, reward, done, _ = env.step(action)

            # Guardar transición
            episode.append((state, action, reward))

            if done:
                break

            state = next_state

        return episode

    def calculate_returns(
        self,
        episode: List[Tuple[Any, int, float]]
    ) -> List[Tuple[Any, float]]:
        """
        Calcula los retornos (returns) para cada estado en el episodio.

        El retorno G_t es la suma descontada de recompensas desde el tiempo t:
        G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...

        Parámetros:
        -----------
        episode : List[Tuple[state, action, reward]]
            Episodio completo

        Retorna:
        --------
        state_returns : List[Tuple[state, return]]
            Lista de (estado, retorno) para cada paso del episodio
        """
        state_returns = []
        G = 0  # Retorno acumulado

        # Procesar episodio en reversa para calcular retornos
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            state_returns.append((state, G))

        # Revertir para tener orden cronológico
        state_returns.reverse()

        return state_returns

    def update_value_function(
        self,
        state_returns: List[Tuple[Any, float]]
    ) -> float:
        """
        Actualiza la función de valor usando los retornos del episodio.

        First-Visit MC: Solo usa la primera aparición de cada estado
        Every-Visit MC: Usa todas las apariciones de cada estado

        Parámetros:
        -----------
        state_returns : List[Tuple[state, return]]
            Lista de estados y sus retornos

        Retorna:
        --------
        max_change : float
            Cambio máximo en V durante esta actualización
        """
        max_change = 0
        visited_states = set()

        for state, G in state_returns:
            # First-Visit: solo procesar primera visita
            if self.method == 'first-visit' and state in visited_states:
                continue

            visited_states.add(state)

            # Guardar retorno observado
            self.returns[state].append(G)
            self.visit_counts[state] += 1

            # Calcular nuevo valor como promedio de retornos
            old_value = self.V[state]
            self.V[state] = np.mean(self.returns[state])

            # Registrar cambio
            max_change = max(max_change, abs(self.V[state] - old_value))

        return max_change

    def evaluate_policy(
        self,
        env: Any,
        policy: Callable[[Any], int],
        num_episodes: int,
        max_steps: int = 1000,
        verbose: bool = True
    ) -> Dict:
        """
        Evalúa una política usando Monte Carlo Prediction.

        Parámetros:
        -----------
        env : Environment
            Entorno episódico
        policy : Callable
            Política a evaluar: policy(state) -> action
        num_episodes : int
            Número de episodios a simular
        max_steps : int
            Máximo de pasos por episodio
        verbose : bool
            Si True, imprime progreso

        Retorna:
        --------
        results : dict
            Diccionario con función de valor y estadísticas
        """
        if verbose:
            print(f"Iniciando MC Prediction ({self.method})...")
            print(f"Episodios: {num_episodes}, Gamma: {self.gamma}\n")

        for episode_num in range(num_episodes):
            # 1. Generar episodio siguiendo la política
            episode = self.generate_episode(env, policy, max_steps)

            # 2. Calcular retornos
            state_returns = self.calculate_returns(episode)

            # 3. Actualizar función de valor
            max_change = self.update_value_function(state_returns)

            # 4. Registrar estadísticas
            episode_return = state_returns[0][1] if state_returns else 0
            self.history['mean_returns'].append(episode_return)
            self.history['value_changes'].append(max_change)
            self.history['states_visited'].append(len(set(s for s, _ in state_returns)))

            # Imprimir progreso
            if verbose and (episode_num + 1) % max(1, num_episodes // 10) == 0:
                avg_return = np.mean(self.history['mean_returns'][-100:])
                num_states = len(self.V)
                print(f"Episodio {episode_num + 1}/{num_episodes}: "
                      f"Return={episode_return:.3f}, "
                      f"Avg Return (últimos 100)={avg_return:.3f}, "
                      f"Estados visitados={num_states}")

        self.history['episodes'] = num_episodes

        if verbose:
            print(f"\nEvaluación completada!")
            print(f"Estados únicos visitados: {len(self.V)}")
            print(f"Visitas promedio por estado: {np.mean(list(self.visit_counts.values())):.1f}")

        return {
            'V': dict(self.V),
            'returns': dict(self.returns),
            'visit_counts': dict(self.visit_counts),
            'history': self.history
        }

    def get_value(self, state: Any) -> float:
        """
        Retorna el valor estimado de un estado.

        Parámetros:
        -----------
        state : Any
            Estado a consultar

        Retorna:
        --------
        value : float
            V^π(s) estimado
        """
        return self.V.get(state, 0.0)

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas del proceso de evaluación.

        Retorna:
        --------
        stats : dict
            Diccionario con estadísticas de entrenamiento
        """
        return {
            'num_states': len(self.V),
            'total_episodes': self.history['episodes'],
            'mean_return': np.mean(self.history['mean_returns']) if self.history['mean_returns'] else 0,
            'std_return': np.std(self.history['mean_returns']) if self.history['mean_returns'] else 0,
            'max_value': max(self.V.values()) if self.V else 0,
            'min_value': min(self.V.values()) if self.V else 0,
            'mean_visits_per_state': np.mean(list(self.visit_counts.values())) if self.visit_counts else 0
        }


# ==================== ENTORNO DE EJEMPLO ====================

class SimpleGridWorld:
    """
    GridWorld simple para demostración de Monte Carlo Prediction.

    Cuadrícula de 5x5 donde el agente debe llegar a la meta.
    - Estado inicial: (0, 0)
    - Estado meta: (4, 4)
    - Recompensa: +1 al llegar a la meta, -0.01 por cada paso
    """

    def __init__(self, size: int = 5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = None

    def reset(self) -> Tuple[int, int]:
        """Reinicia el entorno al estado inicial."""
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Ejecuta una acción en el entorno.

        Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda
        """
        row, col = self.state

        # Aplicar acción
        if action == 0:  # arriba
            row = max(0, row - 1)
        elif action == 1:  # derecha
            col = min(self.size - 1, col + 1)
        elif action == 2:  # abajo
            row = min(self.size - 1, row + 1)
        elif action == 3:  # izquierda
            col = max(0, col - 1)

        self.state = (row, col)

        # Calcular recompensa
        if self.state == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False

        return self.state, reward, done, {}


def create_random_policy(n_actions: int = 4) -> Callable:
    """
    Crea una política aleatoria (uniforme).

    Parámetros:
    -----------
    n_actions : int
        Número de acciones posibles

    Retorna:
    --------
    policy : Callable
        Función de política
    """
    def policy(state):
        return random.randint(0, n_actions - 1)

    return policy


def create_optimistic_policy() -> Callable:
    """
    Crea una política que tiende hacia la meta (derecha y abajo).

    Esta política no es óptima pero es mejor que aleatoria.
    """
    def policy(state):
        row, col = state

        # Con 70% de probabilidad, moverse hacia la meta
        if random.random() < 0.7:
            # Determinar dirección hacia la meta
            if col < 4:  # Preferir derecha
                return 1
            elif row < 4:  # Preferir abajo
                return 2

        # Acción aleatoria
        return random.randint(0, 3)

    return policy


# ==================== ENTORNO BLACKJACK ====================

class SimpleBlackjack:
    """
    Versión simplificada de Blackjack para MC Prediction.

    Estado: (suma_jugador, carta_visible_dealer, tiene_as_utilizable)
    Acciones: 0=plantarse (stick), 1=pedir carta (hit)
    """

    def __init__(self):
        self.state = None
        self.player_sum = 0
        self.dealer_sum = 0
        self.usable_ace = False

    def reset(self) -> Tuple[int, int, bool]:
        """Inicia una nueva mano."""
        # Repartir cartas iniciales
        self.player_sum = random.randint(12, 21)  # Empezar en rango de decisión
        self.dealer_showing = random.randint(1, 10)
        self.usable_ace = random.random() < 0.3  # 30% de tener as utilizable

        self.state = (self.player_sum, self.dealer_showing, self.usable_ace)
        return self.state

    def draw_card(self) -> int:
        """Saca una carta (1-10, donde 1=As)."""
        card = random.randint(1, 13)
        return min(card, 10)

    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """
        Ejecuta una acción.

        0: plantarse (stick)
        1: pedir carta (hit)
        """
        player_sum, dealer_showing, usable_ace = self.state

        if action == 1:  # Hit: pedir carta
            card = self.draw_card()
            player_sum += card

            # Manejar As
            if player_sum > 21 and usable_ace:
                player_sum -= 10
                usable_ace = False

            if player_sum > 21:
                # Jugador se pasa
                return self.state, -1.0, True, {}

            self.state = (player_sum, dealer_showing, usable_ace)
            return self.state, 0.0, False, {}

        else:  # Stick: turno del dealer
            # Simular juego del dealer
            dealer_sum = dealer_showing
            while dealer_sum < 17:
                dealer_sum += self.draw_card()

            # Determinar ganador
            if dealer_sum > 21 or player_sum > dealer_sum:
                reward = 1.0  # Jugador gana
            elif player_sum < dealer_sum:
                reward = -1.0  # Dealer gana
            else:
                reward = 0.0  # Empate

            return self.state, reward, True, {}


def create_blackjack_policy(threshold: int = 20) -> Callable:
    """
    Crea una política simple para Blackjack.

    Pide carta si suma < threshold, se planta si suma >= threshold.
    """
    def policy(state):
        player_sum, dealer_showing, usable_ace = state
        return 1 if player_sum < threshold else 0  # 1=hit, 0=stick

    return policy


# ==================== MAIN ====================

if __name__ == "__main__":
    """
    Ejemplos de uso: Evaluación de políticas con MC Prediction
    """

    print("=" * 70)
    print("MONTE CARLO PREDICTION - Ejemplos")
    print("=" * 70)

    # ========== EJEMPLO 1: GridWorld con First-Visit MC ==========
    print("\n" + "=" * 70)
    print("Ejemplo 1: GridWorld 5x5 - First-Visit MC")
    print("=" * 70)

    # Crear entorno y política
    env1 = SimpleGridWorld(size=5)
    policy1 = create_optimistic_policy()

    # Crear evaluador MC
    mc_first = MCPrediction(gamma=0.99, method='first-visit')

    # Evaluar política
    results1 = mc_first.evaluate_policy(
        env=env1,
        policy=policy1,
        num_episodes=5000,
        max_steps=100,
        verbose=True
    )

    # Mostrar resultados
    print("\nFunción de Valor (algunos estados):")
    states_to_show = [(0, 0), (0, 2), (0, 4), (2, 2), (4, 2), (4, 4)]
    for state in states_to_show:
        if state in mc_first.V:
            print(f"  V{state} = {mc_first.V[state]:.4f} "
                  f"(visitas: {mc_first.visit_counts[state]})")

    stats1 = mc_first.get_statistics()
    print(f"\nEstadísticas:")
    print(f"  - Estados visitados: {stats1['num_states']}")
    print(f"  - Return promedio: {stats1['mean_return']:.4f} ± {stats1['std_return']:.4f}")
    print(f"  - Valor máximo: {stats1['max_value']:.4f}")
    print(f"  - Visitas promedio por estado: {stats1['mean_visits_per_state']:.1f}")

    # ========== EJEMPLO 2: GridWorld con Every-Visit MC ==========
    print("\n" + "=" * 70)
    print("Ejemplo 2: GridWorld 5x5 - Every-Visit MC")
    print("=" * 70)

    # Crear evaluador MC
    mc_every = MCPrediction(gamma=0.99, method='every-visit')

    # Evaluar la misma política
    results2 = mc_every.evaluate_policy(
        env=env1,
        policy=policy1,
        num_episodes=5000,
        max_steps=100,
        verbose=True
    )

    # Comparar con First-Visit
    print("\nComparación First-Visit vs Every-Visit:")
    print(f"{'Estado':<12} {'First-Visit':<12} {'Every-Visit':<12} {'Diferencia':<12}")
    print("-" * 50)
    for state in states_to_show:
        if state in mc_first.V and state in mc_every.V:
            v_first = mc_first.V[state]
            v_every = mc_every.V[state]
            diff = abs(v_first - v_every)
            print(f"{str(state):<12} {v_first:<12.4f} {v_every:<12.4f} {diff:<12.4f}")

    # ========== EJEMPLO 3: Blackjack ==========
    print("\n" + "=" * 70)
    print("Ejemplo 3: Blackjack - Evaluando política conservadora")
    print("=" * 70)

    # Crear entorno y política
    env3 = SimpleBlackjack()
    policy3 = create_blackjack_policy(threshold=18)  # Plantarse en 18

    # Crear evaluador MC
    mc_blackjack = MCPrediction(gamma=1.0, method='first-visit')  # gamma=1.0 para juegos finitos

    # Evaluar política
    results3 = mc_blackjack.evaluate_policy(
        env=env3,
        policy=policy3,
        num_episodes=10000,
        max_steps=50,
        verbose=True
    )

    # Mostrar algunos valores
    print("\nFunción de Valor (estados seleccionados):")
    print(f"{'Estado (suma, dealer, as)':<35} {'Valor':<10} {'Visitas':<10}")
    print("-" * 55)
    sample_states = [
        (20, 10, False), (20, 7, False), (18, 10, False),
        (18, 7, False), (16, 10, False), (14, 7, False)
    ]
    for state in sample_states:
        if state in mc_blackjack.V:
            print(f"{str(state):<35} {mc_blackjack.V[state]:<10.4f} "
                  f"{mc_blackjack.visit_counts[state]:<10}")

    stats3 = mc_blackjack.get_statistics()
    print(f"\nEstadísticas Blackjack:")
    print(f"  - Estados visitados: {stats3['num_states']}")
    print(f"  - Return promedio: {stats3['mean_return']:.4f} ± {stats3['std_return']:.4f}")
    print(f"  - Ratio de victoria estimado: {(stats3['mean_return'] + 1) / 2:.2%}")

    print("\n" + "=" * 70)
    print("Ejemplos completados!")
    print("=" * 70)
