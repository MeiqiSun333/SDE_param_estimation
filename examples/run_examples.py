"""
Example script demonstrating model usage.

Run this script to see the model in action.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.network_model import EmotionNetworkModel
from src.visualization.plots import NetworkVisualizer


def run_single_simulation():
    """Run a single model simulation."""
    
    # Initialize model
    model = EmotionNetworkModel(stochastic=True, random_seed=42)
    print(f"\nModel: {model}")
    
    # Run simulation
    print("\nRunning simulation...")
    time, states = model.simulate(end_time=60)
    print(f"Simulation complete: {states.shape[1]} time steps")
    
    # Visualize
    viz = NetworkVisualizer()
    labels = ['X1: SAD', 'X2: RUM', 'X3: DIST', 'X4: EMOCOPE']
    
    fig = viz.plot_simulation(
        time, states, labels=labels,
        title="Emotion Network Dynamics"
    )
    plt.show()
    
    return time, states


def run_multiple_simulations(n_runs=10):
    """Run multiple simulations to show variability."""
    
    simulations = []
    
    for i in range(n_runs):
        model = EmotionNetworkModel(stochastic=True, random_seed=i)
        time, states = model.simulate(end_time=60)
        simulations.append(states)
        print(f"Run {i+1}/{n_runs} complete")
    
    # Visualize with uncertainty
    viz = NetworkVisualizer()
    labels = ['X1: SAD', 'X2: RUM', 'X3: DIST', 'X4: EMOCOPE']
    
    fig = viz.plot_multiple_simulations(
        time, simulations, labels=labels,
        title=f"Emotion Network Dynamics - {n_runs} Runs"
    )
    plt.show()


def demonstrate_phase_space():
    """Show phase space dynamics."""
    
    model = EmotionNetworkModel(stochastic=True, random_seed=42)
    time, states = model.simulate(end_time=60)
    
    viz = NetworkVisualizer()
    
    # SAD vs RUM
    fig1 = viz.plot_phase_space(
        states,
        state_indices=(0, 1),
        labels=['X1: SAD', 'X2: RUM'],
        title="Phase Space: Sadness vs Rumination"
    )
    
    # DIST vs EMOCOPE
    fig2 = viz.plot_phase_space(
        states,
        state_indices=(2, 3),
        labels=['X3: DIST', 'X4: EMOCOPE'],
        title="Phase Space: Distraction vs Emotional Coping"
    )
    
    plt.show()


def compare_deterministic_stochastic():
    """Compare deterministic and stochastic versions."""

    # Deterministic
    model_det = EmotionNetworkModel(stochastic=False, random_seed=42)
    time_det, states_det = model_det.simulate(end_time=60)
    
    # Stochastic
    model_sto = EmotionNetworkModel(stochastic=True, random_seed=42)
    time_sto, states_sto = model_sto.simulate(end_time=60)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = ['X1: SAD', 'X2: RUM', 'X3: DIST', 'X4: EMOCOPE']
    
    for i, (ax, label) in enumerate(zip(axes.flatten(), labels)):
        ax.plot(time_det, states_det[i, :], label='Deterministic', 
               linewidth=2.5, color='blue')
        ax.plot(time_sto, states_sto[i, :], label='Stochastic', 
               linewidth=2, color='red', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('State Value')
        ax.set_title(label, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
    
    fig.suptitle('Deterministic vs Stochastic Dynamics', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def custom_network_example():
    """Custom network configuration."""
    
    # Define custom network with different parameters
    custom_nodes = [
        {
            "name": "External Stressor",
            "func": "constant",
            "target": 0.7,  # High stress
            "speed": 0.3,
            "sigma": 0.05,
            "inputs": []
        },
        {
            "name": "Negative Thoughts",
            "func": "alogistic",
            "alpha": 8,  # Steep response
            "tau": 0.4,
            "speed": 0.8,  # Fast adaptation
            "sigma": 0.15,
            "inputs": [(0, 0.9)]  # Strong influence from stress
        },
        {
            "name": "Coping Attempt",
            "func": "alogistic",
            "alpha": 5,
            "tau": 0.6,
            "speed": 0.4,
            "sigma": 0.1,
            "inputs": [(1, 0.7)]  # Triggered by negative thoughts
        },
        {
            "name": "Well-being",
            "func": "alogistic",
            "alpha": 5,
            "tau": 0.3,
            "speed": 0.5,
            "sigma": 0.12,
            "inputs": [(1, -0.5), (2, 0.6)]  # Hurt by neg thoughts, helped by coping
        }
    ]
    
    model = EmotionNetworkModel(nodes=custom_nodes, stochastic=True, random_seed=42)
    time, states = model.simulate(end_time=60)
    
    viz = NetworkVisualizer()
    labels = [node["name"] for node in custom_nodes]
    
    fig = viz.plot_simulation(
        time, states, labels=labels,
        title="Custom Network: Stress-Coping-Wellbeing"
    )
    plt.show()


def main():
    """Run all examples."""

    run_single_simulation()
    run_multiple_simulations(n_runs=10)
    demonstrate_phase_space()
    compare_deterministic_stochastic()
    custom_network_example()

if __name__ == "__main__":
    main()
