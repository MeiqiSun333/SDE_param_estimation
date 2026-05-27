"""
Core network model implementation.

Implements a stochastic differential equation-based network model.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class EmotionNetworkModel:
    """
    Stochastic network model.
    
    The model simulates interconnected emotional states (sadness, rumination,
    distraction, emotional coping) using coupled differential equations.
    """
    
    def __init__(
        self,
        nodes: Optional[List[Dict]] = None,
        dt = 0.5,
        stochastic = True,
        random_seed: Optional[int] = None
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.dt = dt
        self.stochastic = stochastic
        self.num_states = 4
        self.end_time = 60
        
        if nodes is None:
            self.nodes = self._get_default_network()
        else:
            self.nodes = nodes
            self.num_states = len(nodes)
    
    def _get_default_network(self):
        return [
            {
                "name": "X1: SAD",
                "func": "constant",
                "target": 0.5,
                "speed": 0.5,
                "sigma": 0.1,
                "inputs": []
            },
            {
                "name": "X2: RUM",
                "func": "alogistic",
                "alpha": 5,
                "tau": 0.3,
                "speed": 0.5,
                "sigma": 0.1,
                "inputs": [(0, 0.7)]
            },
            {
                "name": "X3: DIST",
                "func": "alogistic",
                "alpha": 5,
                "tau": 0.5,
                "speed": 0.5,
                "sigma": 0.1,
                "inputs": [(0, 0.6)]
            },
            {
                "name": "X4: EMOCOPE",
                "func": "alogistic",
                "alpha": 5,
                "tau": 0.4,
                "speed": 0.5,
                "sigma": 0.1,
                "inputs": [(1, -0.3), (2, 0.8)]
            }
        ]
    
    def alogistic(self, impact_sum, alpha, tau):

        numerator = 1 / (1 + np.exp(-alpha * (impact_sum - tau)))
        denominator = 1 / (1 + np.exp(alpha * tau))
        scaling = 1 + np.exp(-alpha * tau)

        return (numerator - denominator) * scaling
    
    def _compute_aggregate_impact(self, node, current_states):
        impact_sum = sum(
            weight * current_states[idx] 
            for idx, weight in node["inputs"]
        )
        
        if node["func"] == "constant":
            return node["target"]
        elif node["func"] == "alogistic":
            return self.alogistic(impact_sum, node["alpha"], node["tau"])

    
    def simulate(
        self, 
        end_time: Optional[float] = None,
        initial_states: Optional[np.ndarray] = None
    ):
        
        if end_time is None:
            end_time = self.end_time
        
        steps = int(end_time / self.dt)
        
        X = np.zeros((self.num_states, steps + 1))
        if initial_states is not None:
            X[:, 0] = initial_states
        
        t = np.linspace(0, end_time, steps + 1)
        
        for k in range(steps):
            current_states = X[:, k]
            next_states = np.zeros(self.num_states)
            
            if self.stochastic:
                dW = np.random.normal(0, np.sqrt(self.dt), self.num_states)
            else:
                dW = np.zeros(self.num_states)
            
            for i, node in enumerate(self.nodes):
                agg_impact = self._compute_aggregate_impact(node, current_states)
                
                # Drift term
                drift = node["speed"] * (agg_impact - current_states[i])
                
                # Diffusion term: Bounded to keep states in [0, 1], exactly matching toy_model.py
                diffusion = 0
                if self.stochastic:
                    diffusion = node["sigma"] * current_states[i] * (1 - current_states[i]) * dW[i]
                
                # Euler-Maruyama update
                next_val = current_states[i] + (drift * self.dt) + diffusion
                next_states[i] = np.clip(next_val, 0, 1)
            
            X[:, k+1] = next_states
        
        return t, X


if __name__ == "__main__":
    model = EmotionNetworkModel(stochastic=True, random_seed=42)
    time, states = model.simulate(end_time=60)
    print(f"Simulation complete: Data shape {states.shape}")
