# SDE_param_estimation

A computational framework for adaptive network modeling with stochastic differential equations.

This project bridges psychological theory and computational modeling by implementing theory-driven network architectures grounded in emotion regulation research. The framework supports parameter estimation from real-world longitudinal data, enabling personalized model fitting and individual-level predictions.

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

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd emotion_network_model

# Install dependencies
pip install -r requirements.txt
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

## 📊 Model Components

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

---

**Built with**: NumPy, Pandas, SciPy, JAX, Matplotlib, Seaborn
