# SDE_param_estimation

A computational framework for adaptive network modeling with stochastic differential equations.

This repository implements a rigorous framework for estimating the parameters of a multidimensional Stochastic Differential Equation (SDE) model. Because our model features state-dependent bounded diffusion (i.e., $\sigma X(1-X)dW$) to accommodate assumptions from psychology, economics and finance, ecology, etc., traditional linear estimation methods are insufficient. We provide several estimation engines, ranging from classical mathematically-derived likelihood methods to modern simulation-based approaches.

This project demonstrates a classic statistical parameter estimation method for SDEs using a theory-driven network architecture based on emotion regulation research as an example. This framework supports parameter estimation from real-world longitudinal data, enabling personalized model fitting and individual-level prediction.

## Overview

This project implements a dynamical systems model to simulate emotional states (sadness, rumination, distraction, emotional coping) and their temporal interactions. The model uses:

- **Theory-driven network topology** based on psychological research on emotion regulation
- **Stochastic differential equations** for realistic variability
- **Parameter estimation** from empirical data using multiple optimization algorithms
- **Empirical data integration** with complete preprocessing pipeline for longitudinal emotion studies

## Project Structure

```
emotion_network_model/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Core model implementations
│   ├── visualization/  # Plotting and analysis tools
│   └── utils/          # Helper functions
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Unit tests
├── config/             # Configuration files
├── data/
│   ├── raw/           # Original datasets
│   └── processed/     # Cleaned data
└── results/           # Model outputs and figures
```

## Installation

Clone the repository and install the required dependencies. Note that this project heavily relies on JAX for automatic differentiation.

```bash
git clone [https://github.com/yourusername/SDE_param_estimation.git](https://github.com/yourusername/SDE_param_estimation.git)
cd SDE_param_estimation
pip install -r requirements.txt
```

(Note: For JAX GPU support, please follow [the official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).)

## Quick Start & Validation

We provide a comprehensive test suite to demonstrate the accuracy of our parameter recovery engines. You can run the optimization test to see how well the algorithms recover parameters from high-frequency synthetic SDE data:

```bash
python test_optimization.py
```

Expected Output:
The script will generate perfect transition data and output a comparison table like this:

```Plaintext
========================================================
  PARAMETER RECOVERY REPORT: AIT_SAHALIA
========================================================
Parameter       | True Value   | Estimated    | Abs Error   
------------------------------------------------------------
Beta_1          | 0.4000       | 0.4012       | 0.0012      
Beta_2          | 0.6000       | 0.5985       | 0.0015      
...
Sigma_1         | 0.1500       | 0.1503       | 0.0003      
========================================================
```

To run the forward simulations and visualize the emotion network dynamics:

```Bash
python run_examples.py
```

### Basic Usage

```python
from src.models.network_model import EmotionNetworkModel

# Initialize model with default parameters
model = EmotionNetworkModel(stochastic=True, random_seed=42)

# Run simulation
time, states = model.simulate(end_time=60)

# Visualize results
model.plot_results(time, states)
```

### Run Full Pipeline

```bash
# Data preprocessing
python -m src.data.preprocessing

# Model training and evaluation
python -m src.models.train

# Generate visualizations
python -m src.visualization.plots
```

## Model Components

### State Variables
- **X1 (SAD)**: Sadness level with fluctuating baseline
- **X2 (RUM)**: Rumination influenced by sadness
- **X3 (DIST)**: Distraction capability
- **X4 (EMOCOPE)**: Emotional coping effectiveness

### Key Features
- Flexible network topology configuration
- Multiple combination functions (logistic, step, baseline)
- Stochastic processes with configurable noise
- Parameter optimization with gradient-based methods
- Comprehensive visualization suite

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Results

The model successfully captures:
- Non-linear emotion dynamics
- State-dependent transitions
- Realistic temporal patterns
- Individual variability through stochastic components

## Parameter Estimation Methods (Likelihood-Based)
These methods rely on deriving closed-form approximations of the transition density and using Autograd (JAX) and L-BFGS-B to maximize the log-likelihood.

* **Euler-Maruyama (EM) Approximation**
  * **Pros:** Easy to implement and computationally extremely fast. Serves as a standard baseline.
  * **Cons:** Only achieves 0.5 order of strong convergence. It assumes the diffusion coefficient remains constant over the time step $\Delta t$, which can lead to biased estimates if the sampling interval is large.

* **Milstein Scheme (Pseudo-Gaussian)**
  * **Pros:** Achieves 1.0 order of strong convergence by incorporating the derivative of the diffusion term via Itô's Lemma. Highly suitable for our state-dependent volatility.
  * **Cons:** Harder to formulate analytically for multi-dimensional coupled networks. Using a Pseudo-Gaussian approximation for the likelihood ignores the true non-Gaussian heavy tails of the Milstein steps.

* **Aït-Sahalia Polynomial Expansion**
  * **Pros:** Highly accurate even for sparse data (large $\Delta t$). By applying the Lamperti transform to stabilize the variance and calculating the Taylor expansion of the transition density, it mathematically approximates the *true* continuous-time dynamics.
  * **Cons:** Extremely complex to derive and implement. The analytical formulas are highly specific to the model's structural equations, meaning any change to the network requires re-deriving the Jacobian, Hessian, and integration formulas.

## Alternative Approaches: Likelihood-Free / Simulation-Based Methods (Not implemented in this repository)
For highly complex networks where deriving the exact transition density is intractable, we can treat the SDE solver as a "black box" and rely on computational power and machine learning.

* **Approximate Bayesian Computation (ABC)**
  * **Pros:** "Zero math" required. It simply simulates forward paths with random parameters and keeps the ones that generate data closely matching the real dataset. Highly intuitive.
  * **Cons:** Computationally expensive and suffers heavily from the "curse of dimensionality." Finding 12+ optimal parameters (betas, sigmas, weights) through pure random sampling is inefficient.

* **Simulated Maximum Likelihood (SML)**
  * **Pros:** Uses Monte Carlo simulations (thousands of short paths) combined with Kernel Density Estimation (KDE) to empirically construct the transition probability, bypassing complex calculus. 
  * **Cons:** Sensitive to simulation variance. High-dimensional KDE scales poorly, requiring massive parallel simulations (e.g., via JAX `vmap`) to maintain accuracy.

* **Simulation-Based Inference (SBI / Neural Density Estimation)**
  * **Pros:** The state-of-the-art AI approach. Uses deep learning (e.g., Normalizing Flows) to learn the inverse mapping from simulated time-series features to the posterior parameter distribution. Once trained, evaluating new datasets is instantaneous.
  * **Cons:** Requires setting up a PyTorch/neural network pipeline. The inference acts as a black box, offering less mathematical interpretability than classical analytical methods.

---

**Built with**: NumPy, Pandas, SciPy, JAX, Matplotlib, Seaborn
