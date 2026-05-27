"""
Parameter estimation and optimization.

Implements Maximum Likelihood Estimation (MLE) using:
1. Euler-Maruyama (EM) transition density approximation.
2. Aït-Sahalia closed-form transition density polynomial expansion.

This script leverages JAX for automatic differentiation (Autograd) and 
Just-In-Time (JIT) compilation to bridge with SciPy's L-BFGS-B optimizer.

Notes: The measurement of loss may differ between models,
especially for high-dimensional SDE parameter estimation.
MSE used here might be not suitable for this scenario.
Consider Simulated Method of Moments, Simulation-Based Inference,
or Maximum Likelihood Estimation using transition probabilities.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, jacfwd, hessian
jax.config.update("jax_enable_x64", True)


class SDEModelCore:
    """
    Core mathematical definitions for the specific Emotion Network SDE.
    """
    mu_fixed = 0.5
    alphas_fixed = jnp.array([5, 5, 5])
    taus_fixed = jnp.array([0.3, 0.5, 0.4])

    @staticmethod
    def alogistic_jax(impact: jnp.ndarray, alpha, tau) -> jnp.ndarray:
        num = 1 / (1 + jnp.exp(-alpha * (impact - tau)))
        den = 1 / (1 + jnp.exp(alpha * tau))
        scaling = 1 + jnp.exp(-alpha * tau)
        return (num - den) * scaling

    @staticmethod
    def compute_S(Y: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Computes the aggregate impact S for the 4 states."""
        w21, w31, w42, w43 = weights
        y1, y2, y3 = Y[0], Y[1], Y[2]
        
        s1 = SDEModelCore.mu_fixed 
        s2 = SDEModelCore.alogistic_jax(w21 * y1, SDEModelCore.alphas_fixed[0], SDEModelCore.taus_fixed[0])
        s3 = SDEModelCore.alogistic_jax(w31 * y1, SDEModelCore.alphas_fixed[1], SDEModelCore.taus_fixed[1])
        s4 = SDEModelCore.alogistic_jax((w42 * y2) + (w43 * y3), SDEModelCore.alphas_fixed[2], SDEModelCore.taus_fixed[2])
        
        return jnp.array([s1, s2, s3, s4])

    @staticmethod
    def mu_Z_fn(z: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Drift in the Lamperti-transformed Z space."""
        betas = params[0:4]
        sigmas = params[4:8]
        weights = params[8:12]
        
        # Inverse Lamperti
        Y = 1 / (1 + jnp.exp(-sigmas * z))
        S = SDEModelCore.compute_S(Y, weights)
        
        drift_z = (betas * (S - Y)) / (sigmas * Y * (1 - Y)) - (sigmas * (1 - 2 * Y) / 2)
        return drift_z


class SDEEstimators:
    """
    Log-Likelihood functions for different SDE approximation methods.
    """
    
    @staticmethod
    def euler_maruyama_logpdf(params: jnp.ndarray, Y_prev: jnp.ndarray, 
                              Y_curr: jnp.ndarray, dt):
        betas = params[0:4]
        sigmas = params[4:8]
        weights = params[8:12]
        
        S = SDEModelCore.compute_S(Y_prev, weights)
        
        drift = betas * (S - Y_prev)
        diffusion = sigmas * Y_prev * (1 - Y_prev)
        
        variance = (diffusion ** 2) * dt + 1e-10
        residual = Y_curr - Y_prev - drift * dt
        
        ll = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + jnp.log(variance) + (residual ** 2) / variance)

        return ll
    
    @staticmethod
    def milstein_logpdf(params: jnp.ndarray, Y_prev: jnp.ndarray, 
                        Y_curr: jnp.ndarray, dt: float) -> jnp.ndarray:
        betas = params[0:4]
        sigmas = params[4:8]
        weights = params[8:12]
        
        S = SDEModelCore.compute_S(Y_prev, weights)

        drift = betas * (S - Y_prev)
        
        diffusion = sigmas * Y_prev * (1 - Y_prev)
        diffusion_prime = sigmas * (1 - 2 * Y_prev)
        
        # Milstein expectation: X_t + a(X_t)*dt
        mean = Y_prev + drift * dt
        
        # Milstein variance: b^2 * dt + 0.5 * (b * b')^2 * dt^2
        variance = (diffusion ** 2) * dt + 0.5 * (diffusion * diffusion_prime) ** 2 * (dt ** 2) + 1e-10
        
        residual = Y_curr - mean
        
        ll = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + jnp.log(variance) + (residual ** 2) / variance)
        return ll

    # Helper functions for Aït-Sahalia Integration
    @staticmethod
    def _C0_fn(z: jnp.ndarray, z0: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        def integrand(u):
            zu = z0 + u * (z - z0)
            return jnp.dot(SDEModelCore.mu_Z_fn(zu, params), (z - z0))
        u_vals = jnp.linspace(0, 1, 10)
        vals = vmap(integrand)(u_vals)
        return jnp.trapezoid(vals, u_vals)

    @staticmethod
    def _C1_fn(z: jnp.ndarray, z0: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        grad_C0_fn = jax.grad(SDEEstimators._C0_fn, argnums=0)
        lap_C0_fn = lambda z_val, z0_val, p: jnp.trace(hessian(SDEEstimators._C0_fn, argnums=0)(z_val, z0_val, p))
        div_mu_fn = lambda z_val, p: jnp.trace(jacfwd(SDEModelCore.mu_Z_fn, argnums=0)(z_val, p))
        
        def G1_fn(z_val, z0_val, p):
            div = div_mu_fn(z_val, p)
            grad_C0 = grad_C0_fn(z_val, z0_val, p)
            lap_C0 = lap_C0_fn(z_val, z0_val, p)
            mu_val = SDEModelCore.mu_Z_fn(z_val, p)
            return -div - jnp.dot(grad_C0, mu_val) + 0.5 * lap_C0 + 0.5 * jnp.sum(grad_C0**2)

        def integrand(u):
            zu = z0 + u * (z - z0)
            return G1_fn(zu, z0, params)
            
        u_vals = jnp.linspace(0, 1, 10)
        vals = vmap(integrand)(u_vals)
        return jnp.trapezoid(vals, u_vals)

    @staticmethod
    def ait_sahalia_logpdf(params: jnp.ndarray, Y_prev: jnp.ndarray, 
                           Y_curr: jnp.ndarray, dt: float) -> jnp.ndarray:
        sigmas = params[4:8]
        
        # Lamperti transform
        Z_prev = (1 / sigmas) * jnp.log(Y_prev / (1 - Y_prev))
        Z_curr = (1 / sigmas) * jnp.log(Y_curr / (1 - Y_curr))
        
        delta_Z = Z_curr - Z_prev
        C_minus1 = -0.5 * jnp.sum(delta_Z**2)
        
        c0 = SDEEstimators._C0_fn(Z_curr, Z_prev, params)
        c1 = SDEEstimators._C1_fn(Z_curr, Z_prev, params)
        
        l_Z = -2 * jnp.log(2 * jnp.pi * dt) + (C_minus1 / dt) + c0 + c1 * dt 
        log_jac = jnp.sum(-jnp.log(sigmas) - jnp.log(Y_curr) - jnp.log(1 - Y_curr))
        
        return l_Z + log_jac


class JAXOptimizer:
    """
    Robust JAX-based optimization engine bridging JAX gradients.
    """
    def __init__(self, method: str = 'euler_maruyama'):
        if method == 'euler_maruyama':
            self.core_pdf = SDEEstimators.euler_maruyama_logpdf
        elif method == 'milstein':
            self.core_pdf = SDEEstimators.milstein_logpdf
        elif method == 'ait_sahalia':
            self.core_pdf = SDEEstimators.ait_sahalia_logpdf

            
        # Vectorize across the batch dimension (N transitions)
        self.batch_pdf = vmap(self.core_pdf, in_axes=(None, 0, 0, 0))
        self.val_and_grad_fn = jit(value_and_grad(self._negative_log_likelihood))

    def _negative_log_likelihood(self, params: jnp.ndarray, Y_prev: jnp.ndarray, 
                                 Y_curr: jnp.ndarray, dts: jnp.ndarray) -> jnp.ndarray:
        log_likelihoods = self.batch_pdf(params, Y_prev, Y_curr, dts)
        
        # Weight penalty matching your toy_model.py
        penalty = 0.05 * jnp.sum(params[8:12]**2) 
        
        # We sum the negative log-likelihoods (or mean if you prefer scaling)
        return -jnp.sum(log_likelihoods) + penalty

    def _scipy_wrapper(self, params: np.ndarray, *args):
        val, grad = self.val_and_grad_fn(params, *args)
        return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)

    def optimize(self, initial_params, Y_prev, Y_curr, dts, bounds, max_iter = 200):
                 
        result = minimize(
            self._scipy_wrapper,
            initial_params,
            args=(Y_prev, Y_curr, dts),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )
        return result.x, result.fun, result.success


def run_estimation(Y_prev_all, Y_curr_all, dt_all, method = 'euler_maruyama'):
    
    print(f"\nStarting {method.upper()} estimation engine")
    
    # 12 Parameters: 4 Betas, 4 Sigmas, 4 Weights
    init_params = np.array([
        0.5, 0.5, 0.5, 0.5,  # betas
        0.2, 0.2, 0.2, 0.2,  # sigmas
        0.1, 0.1, 0.1, 0.1   # weights
    ])


    bounds = ([(0.001, 5.0)] * 4 + [(0.01, 5.0)] * 4 + [(-5.0, 5.0)] * 4)
              
    optimizer = JAXOptimizer(method=method)
    
    Y_prev_jax = jnp.array(Y_prev_all)
    Y_curr_jax = jnp.array(Y_curr_all)
    dt_jax = jnp.array(dt_all)
    
    opt_params, final_loss, success = optimizer.optimize(
        initial_params=init_params,
        Y_prev=Y_prev_jax,
        Y_curr=Y_curr_jax,
        dts=dt_jax,
        bounds=bounds
    )
    
    if success:
        print("Estimation Success!")
    else:
        print("Estimation Stopped (check optimization status).")
        
    print(f"Betas:   {opt_params[0:4]}")
    print(f"Sigmas:  {opt_params[4:8]}")
    print(f"Weights: {opt_params[8:12]}")
    
    return opt_params


if __name__ == "__main__":
    
    file_path = 'data/processed/emotional_data_processed.csv'
    target_columns = ['UUID', 'time_hours', 'SAD', 'RUM', 'DIST', 'EMOCOPE', 'delta_t']
    

    df = pd.read_csv(file_path, usecols=target_columns).dropna()


    feature_columns = ['SAD', 'RUM', 'DIST', 'EMOCOPE']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    

    epsilon = 1e-5
    for col in feature_columns:
        df[col] = df[col].clip(epsilon, 1 - epsilon)

    Y_prev_list, Y_curr_list, dt_list = [], [], []
    

    for uuid, group in df.groupby('UUID'):
        group = group.sort_values('time_hours')
        data_Y = group[feature_columns].values
        delta_t_array = group['delta_t'].values[1:]
        
        Y_prev_list.append(data_Y[:-1])
        Y_curr_list.append(data_Y[1:])
        dt_list.append(delta_t_array)

    Y_prev_all = np.vstack(Y_prev_list)
    Y_curr_all = np.vstack(Y_curr_list)
    dt_all = np.concatenate(dt_list)
    

    estimated_parameters = run_estimation(Y_prev_all, Y_curr_all, dt_all, method='euler_maruyama')