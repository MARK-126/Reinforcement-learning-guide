"""
Experience Replay Buffer

Buffer para almacenar y muestrear transiciones de forma eficiente.
Usado en algoritmos off-policy como DQN, DDPG, SAC, etc.
"""

import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional


# Transición: (estado, acción, recompensa, siguiente estado, done)
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Simple Experience Replay Buffer
    
    Almacena transiciones (s, a, r, s', done) en un buffer circular.
    Permite muestreo aleatorio uniforme.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Tamaño máximo del buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Añade una transición al buffer
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio terminó
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """
        Muestrea un batch aleatorio de transiciones
        
        Args:
            batch_size: Tamaño del batch
        
        Returns:
            Lista de transiciones
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Retorna el número de transiciones en el buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Limpia el buffer"""
        self.buffer.clear()


class NumpyReplayBuffer:
    """
    Replay Buffer optimizado usando NumPy arrays
    
    Más eficiente en memoria y velocidad que usar deque.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1):
        """
        Args:
            capacity: Tamaño máximo del buffer
            state_dim: Dimensión del espacio de estados
            action_dim: Dimensión del espacio de acciones (1 para discreto)
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.position = 0
        self.size = 0
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        """Añade transición al buffer"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Muestrea batch de transiciones
        
        Returns:
            Tupla de (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        return self.size
    
    def clear(self):
        """Limpia el buffer"""
        self.position = 0
        self.size = 0


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Muestrea transiciones con probabilidad proporcional a su TD error.
    Paper: "Prioritized Experience Replay" (Schaul et al., 2015)
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        """
        Args:
            capacity: Tamaño máximo del buffer
            alpha: Cuánto usar prioridad (0 = uniforme, 1 = full prioritization)
            beta: Importance sampling correction (0 = no correction, 1 = full)
            beta_increment: Incremento de beta por sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done, error: Optional[float] = None):
        """
        Añade transición con prioridad
        
        Args:
            error: TD error (si None, usa prioridad máxima)
        """
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Transition(state, action, reward, next_state, done)
        
        # Asignar prioridad
        self.priorities[self.position] = max_priority if error is None else abs(error) + 1e-6
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Muestrea batch con prioridades
        
        Returns:
            transitions: Lista de transiciones
            indices: Índices en el buffer (para actualizar prioridades)
            weights: Importance sampling weights
        """
        # Calcular probabilidades
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Muestrear
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        transitions = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalizar
        
        # Incrementar beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """
        Actualiza prioridades basado en nuevos TD errors
        
        Args:
            indices: Índices de las transiciones
            errors: Nuevos TD errors
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self) -> int:
        return self.size


class EpisodeReplayBuffer:
    """
    Buffer que almacena episodios completos
    
    Útil para algoritmos que necesitan secuencias completas (RNNs, etc.)
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Args:
            capacity: Número máximo de episodios
        """
        self.episodes = deque(maxlen=capacity)
    
    def push_episode(self, episode: List[Transition]):
        """
        Añade episodio completo
        
        Args:
            episode: Lista de transiciones del episodio
        """
        self.episodes.append(episode)
    
    def sample_episodes(self, n_episodes: int) -> List[List[Transition]]:
        """
        Muestrea episodios completos
        
        Args:
            n_episodes: Número de episodios a muestrear
        
        Returns:
            Lista de episodios
        """
        return random.sample(self.episodes, n_episodes)
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """
        Muestrea transiciones individuales de episodios aleatorios
        
        Args:
            batch_size: Número de transiciones
        
        Returns:
            Lista de transiciones
        """
        transitions = []
        while len(transitions) < batch_size:
            episode = random.choice(self.episodes)
            transition = random.choice(episode)
            transitions.append(transition)
        return transitions
    
    def __len__(self) -> int:
        return len(self.episodes)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear buffer simple
    buffer = ReplayBuffer(capacity=1000)
    
    # Añadir transiciones
    for i in range(100):
        state = np.random.rand(4)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.rand(4)
        done = i % 10 == 0
        
        buffer.push(state, action, reward, next_state, done)
    
    # Muestrear batch
    batch = buffer.sample(32)
    print(f"Buffer size: {len(buffer)}")
    print(f"Batch size: {len(batch)}")
    print(f"Sample transition: {batch[0]}")
    
    # Ejemplo con NumpyReplayBuffer
    np_buffer = NumpyReplayBuffer(capacity=1000, state_dim=4, action_dim=1)
    
    for i in range(100):
        state = np.random.rand(4)
        action = np.array([np.random.randint(2)])
        reward = np.random.randn()
        next_state = np.random.rand(4)
        done = i % 10 == 0
        
        np_buffer.push(state, action, reward, next_state, done)
    
    # Muestrear
    states, actions, rewards, next_states, dones = np_buffer.sample(32)
    print(f"\nNumPy buffer size: {len(np_buffer)}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
