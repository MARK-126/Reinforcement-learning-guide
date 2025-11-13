"""
Monte Carlo Methods
====================

Métodos Monte Carlo para RL:
- MC Prediction (Predicción Monte Carlo)
- MC Control (Control Monte Carlo)

Clases disponibles:
- MCPrediction: Evaluación de políticas con Monte Carlo (First-Visit y Every-Visit)
- MCControlOnPolicy: Control On-Policy con ε-greedy
- MCControlOffPolicy: Control Off-Policy con Importance Sampling
"""

from .mc_prediction import MCPrediction, SimpleGridWorld, SimpleBlackjack
from .mc_control import MCControlOnPolicy, MCControlOffPolicy, CliffWalking

__all__ = [
    "MCPrediction",
    "MCControlOnPolicy",
    "MCControlOffPolicy",
    "SimpleGridWorld",
    "SimpleBlackjack",
    "CliffWalking"
]
