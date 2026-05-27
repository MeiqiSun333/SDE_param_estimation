"""
Optimization Test.

Generates high-frequency perfect synthetic data based on the target SDE equations,
and runs the parameter estimation engines to verify accurate parameter recovery.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from src.models.optimization import run_estimation

# Fixed structural hyper-parameters matching SDEModelCore
mu_fixed = 0.5
alpha_fixed = np.array([5, 5, 5])
taus_fixed = np.array([0.3, 0.5, 0.4])


def alogistic(impact, alpha, tau):
    """Numpy implementation of the normalized sigmoid function."""
    num = 1 / (1 + np.exp(-alpha * (impact - tau)))
    den = 1 / (1 + np.exp(alpha * tau))
    scaling = 1 + np.exp(-alpha * tau)
    return (num - den) * scaling


def generate_perfect_data(n_trajectories = 25, 
    steps_per_traj = 120, 
    dt = 0.01,
    seed = 42):
    """
    Generates high-frequency perfect simulation data matching the specific SDE network.
    """
    np.random.seed(seed)
    
    # Ground Truth Parameters to recover
    true_betas = np.array([0.4, 0.6, 0.3, 0.5])
    true_sigmas = np.array([0.15, 0.20, 0.18, 0.25])
    true_weights = np.array([0.8, 0.5, -0.4, 0.7])  # w21, w31, w42, w43
    
    Y_prev_list, Y_curr_list = [], []
    
    for _ in range(n_trajectories):
        # Initialize randomly within reasonable boundaries
        X = np.random.uniform(0.2, 0.8, 4)
        
        for _ in range(steps_per_traj):
            X_prev = X.copy()
            dW = np.random.normal(0, np.sqrt(dt), 4)
            
            # Compute network aggregate impacts
            s1 = mu_fixed
            s2 = alogistic(true_weights[0] * X_prev[0], alpha_fixed[0], taus_fixed[0])
            s3 = alogistic(true_weights[1] * X_prev[1], alpha_fixed[1], taus_fixed[1])
            s4 = alogistic((true_weights[2] * X_prev[1]) + (true_weights[3] * X_prev[2]), alpha_fixed[2], taus_fixed[2])
            S = np.array([s1, s2, s3, s4])
            
            # Step forward via exact continuous equations
            drift = true_betas * (S - X_prev)
            diffusion = true_sigmas * X_prev * (1 - X_prev)
            
            X_next = X_prev + drift * dt + diffusion * dW
            X_next = np.clip(X_next, 1e-5, 1 - 1e-5)
            
            Y_prev_list.append(X_prev)
            Y_curr_list.append(X_next)
            X = X_next

    return (
        np.array(Y_prev_list), 
        np.array(Y_curr_list), 
        np.full(len(Y_prev_list), dt),
        true_betas, 
        true_sigmas, 
        true_weights
    )


def print_comparison_table(method_name, true_p, est_p):
    """
    Prints a formatted table comparing true vs estimated parameters.
    """

    param_names = (
        [f"Beta_{i+1}" for i in range(4)] + 
        [f"Sigma_{i+1}" for i in range(4)] + 
        ["Weight_21", "Weight_31", "Weight_42", "Weight_43"]
    )
    

    print(f"Parameter estimation report: {method_name.upper()}")
    print(f"{'Parameter':<15} | {'True Value':<12} | {'Estimated':<12} | {'Abs Error':<12}")
    
    for name, t_val, e_val in zip(param_names, true_p, est_p):
        error = abs(t_val - e_val)
        print(f"{name:<15} | {t_val:<12.4f} | {e_val:<12.4f} | {error:<12.4f}")


def main():
    print("Initializing high-frequency data generation generator...")
    # Generate perfect simulated data
    Y_prev, Y_curr, dts, t_betas, t_sigmas, t_weights = generate_perfect_data(
        n_trajectories=30, 
        steps_per_traj=150, 
        dt=0.005
    )
    
    # Concatenate true parameters into a single flat array for comparison
    true_parameters = np.concatenate([t_betas, t_sigmas, t_weights])
    
    print(f"Generated {len(Y_prev)} perfect data transition pairs.")
    
    # Test methods sequentially
    test_methods = ['euler_maruyama', 'milstein', 'ait_sahalia']
    
    for method in test_methods:
        estimated_params = run_estimation(Y_prev, Y_curr, dts, method=method)
        print_comparison_table(method, true_parameters, estimated_params)


if __name__ == "__main__":
    main()