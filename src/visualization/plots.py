"""
Visualization tools.

Provides plotting functions for simulation, EDA, and model diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from pathlib import Path

from src.models.network_model import EmotionNetworkModel

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class NetworkVisualizer:
    """
    Visualization suite.
    """
    
    def __init__(self, style = "whitegrid", figsize = (12, 6)):
        """
        Initialize visualizer.
        
        Args:
            style: Seaborn style
            figsize: Default figure size
        """
        sns.set_style(style)
        self.figsize = figsize
    
    def plot_simulation(
        self, time, states,
        labels: Optional[List[str]] = None,
        title = "Network Simulation",
        save_path: Optional[str] = None):
        """
        Plot simulation trajectories.
        
        Args:
            time: Time array
            states: States array (num_states x time_steps)
            labels: State labels
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if labels is None:
            labels = [f"X{i+1}" for i in range(states.shape[0])]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i in range(states.shape[0]):
            ax.plot(time, states[i, :], label=labels[i], linewidth=2, alpha=0.8)
        
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("State Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        # ax.grid(True, alpha=0.3)
        # ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_simulations(
        self, time, simulations,
        labels: Optional[List[str]] = None,
        title = "Multiple Simulation Runs",
        save_path: Optional[str] = None
    ):
        """
        Plot multiple simulation runs with uncertainty bands.
        
        Args:
            time: Time array
            simulations: List of state arrays
            labels: State labels
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Figure
        """
        if labels is None:
            labels = [f"X{i+1}" for i in range(simulations[0].shape[0])]
        
        # Stack simulations
        sims_array = np.stack(simulations, axis=-1)  # (states, time, runs)
        mean_traj = np.mean(sims_array, axis=-1)
        std_traj = np.std(sims_array, axis=-1)
        
        num_states = sims_array.shape[0]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i in range(num_states):
            ax = axes[i]
            
            for run in range(sims_array.shape[2]):
                ax.plot(time, sims_array[i, :, run], alpha=0.2, color='gray', linewidth=0.5)
            
            # Plot mean trajectory
            ax.plot(time, mean_traj[i, :], label='Mean', linewidth=2.5, color='darkblue')
            
            # Plot uncertainty band
            ax.fill_between(
                time,
                mean_traj[i, :] - std_traj[i, :],
                mean_traj[i, :] + std_traj[i, :],
                alpha=0.3,
                color='skyblue',
                label='±1 SD'
            )
            
            ax.set_xlabel("Time", fontsize=11)
            ax.set_ylabel("State Value", fontsize=11)
            ax.set_title(labels[i], fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            # ax.grid(True, alpha=0.3)
            # ax.set_ylim([-0.05, 1.05])
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_phase_space(
        self, states,
        state_indices = (0, 1),
        labels: Optional[List[str]] = None,
        title = "Phase Space",
        save_path: Optional[str] = None
    ):
        """
        Plot 2D phase space trajectory.
        
        Args:
            states: States array
            state_indices: Which two states to plot (x, y)
            labels: State labels
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        if labels is None:
            labels = [f"X{i+1}" for i in state_indices]
        
        idx_x, idx_y = state_indices
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot trajectory with color gradient
        points = np.array([states[idx_x, :], states[idx_y, :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidth=2)
        lc.set_array(np.linspace(0, 1, len(segments)))
        line = ax.add_collection(lc)
        
        # Mark start and end
        ax.plot(states[idx_x, 0], states[idx_y, 0], 'go', markersize=10, label='Start')
        ax.plot(states[idx_x, -1], states[idx_y, -1], 'ro', markersize=10, label='End')
        
        ax.set_xlabel(labels[0], fontsize=12)
        ax.set_ylabel(labels[1], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        # ax.grid(True, alpha=0.3)
        # ax.set_xlim([-0.05, 1.05])
        # ax.set_ylim([-0.05, 1.05])
        
        fig.colorbar(line, ax=ax, label='Time')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_eda_overview(
        self, df, emotion_cols,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive EDA overview plot.
        
        Args:
            df: real-world dataframe
            emotion_cols: Columns to analyze
            save_path: Optional save path
            
        Returns:
            Figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Observations per subject
        ax1 = fig.add_subplot(gs[0, 0])
        obs_counts = df.groupby('UUID').size()
        ax1.hist(obs_counts, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Observations per Subject')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Observation Counts', fontweight='bold')
        ax1.axvline(obs_counts.mean(), color='red', linestyle='--', label=f'Mean: {obs_counts.mean():.1f}')
        ax1.legend()
        
        # Time series length
        ax2 = fig.add_subplot(gs[0, 1])
        if 'time_hours' in df.columns:
            max_times = df.groupby('UUID')['time_hours'].max()
            ax2.hist(max_times, bins=30, edgecolor='black', alpha=0.7, color='orange')
            ax2.set_xlabel('Study Duration (hours)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Time Series Duration', fontweight='bold')
        
        # Correlation heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        corr = df[emotion_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   ax=ax3, cbar_kws={'shrink': 0.8})
        ax3.set_title('Indices Correlations', fontweight='bold')
        
        # Distributions of each emotion
        for i, col in enumerate(emotion_cols[:3]):
            ax = fig.add_subplot(gs[1, i])
            df[col].hist(bins=30, edgecolor='black', alpha=0.7, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{col} Distribution', fontweight='bold')
        
        # Time series examples
        sample_subjects = df['UUID'].unique()[:3]
        for i, uuid in enumerate(sample_subjects):
            ax = fig.add_subplot(gs[2, i])
            subject_data = df[df['UUID'] == uuid]
            
            if 'time_hours' in subject_data.columns:
                for col in emotion_cols[:4]:
                    if col in subject_data.columns:
                        ax.plot(subject_data['time_hours'], subject_data[col], 
                               label=col, alpha=0.7, marker='o', markersize=2)
                
                ax.set_xlabel('Time (hours)')
                ax.set_ylabel('Value')
                ax.set_title(f'Subject {i+1} Time Series', fontweight='bold')
                ax.legend(fontsize=8)
                # ax.grid(True, alpha=0.3)
        
        fig.suptitle('Emotion Data: Exploratory Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    
    # Initialize model and run simulation
    model = EmotionNetworkModel(stochastic=True, random_seed=42)
    time, states = model.simulate(end_time=60)
    
    # visualization
    viz = NetworkVisualizer()
    
    labels = ['X1: SAD', 'X2: RUM', 'X3: DIST', 'X4: EMOCOPE']
    fig = viz.plot_simulation(time, states, labels=labels, 
                             save_path='results/simulation.png')
    plt.show()


if __name__ == "__main__":
    main()
