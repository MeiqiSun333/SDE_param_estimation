"""
Configuration loading utilities.

Load model parameters and settings from configuration files.
"""

import configparser
from pathlib import Path
from typing import Optional


class ConfigLoader:
    """
    Load and parse configuration files for the model.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to config file (default: config/default_config.ini)
        """
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "default_config.ini")
        
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
    
    def get_simulation_params(self):
        """
        Get simulation parameters.
        
        Returns:
            Dictionary of simulation settings
        """
        section = self.config['simulation']
        
        return {
            'dt': float(section.get('dt', 0.5)),
            'end_time': float(section.get('end_time', 60)),
            'stochastic': section.getboolean('stochastic', True),
            'random_seed': self._parse_optional_int(section.get('random_seed'))
        }
    
    def get_network_topology(self):
        """
        Get network topology settings.
        
        Returns:
            Dictionary of network configuration
        """
        section = self.config['network_topology']
        
        return {
            'num_states': int(section.get('num_states', 4)),
            'state_names': [s.strip() for s in section.get('state_names', 'X1,X2,X3,X4').split(',')]
        }
    
    def get_default_parameters(self):
        """
        Get default model parameters.
        
        Returns:
            Dictionary of parameters
        """
        section = self.config['default_parameters']
        
        return {
            'speeds': self._parse_list(section.get('speeds', '0.5, 0.5, 0.5, 0.5')),
            'sigmas': self._parse_list(section.get('sigmas', '0.1, 0.1, 0.1, 0.1')),
            'weights': {
                0: self._parse_list(section.get('weights_0', '1.0')),
                1: self._parse_list(section.get('weights_1', '0.7')),
                2: self._parse_list(section.get('weights_2', '0.6')),
                3: self._parse_list(section.get('weights_3', '-0.3, 0.8'))
            }
        }
    
    def get_combination_functions(self):
        """
        Get combination function configurations.
        
        Returns:
            Dictionary of function types and parameters
        """
        section = self.config['combination_functions']
        
        num_states = self.get_network_topology()['num_states']
        
        funcs = {}
        params = {}
        
        for i in range(num_states):
            funcs[i] = section.get(f'func_{i}', 'alogistic')
            params[i] = self._parse_list(section.get(f'params_{i}', '5, 0.5'))
        
        return {
            'functions': funcs,
            'params': params
        }
    
    def get_optimization_params(self):
        """
        Get optimization parameters.
        
        Returns:
            Dictionary of optimization settings
        """
        section = self.config['optimization']
        
        return {
            'method': section.get('method', 'scipy'),
            'max_iter': int(section.get('max_iter', 100)),
            'popsize': int(section.get('popsize', 15)),
            'learning_rate': float(section.get('learning_rate', 0.01))
        }
    
    def get_visualization_params(self):
        """
        Get visualization parameters.
        
        Returns:
            Dictionary of visualization settings
        """
        section = self.config['visualization']
        
        figsize_str = section.get('figsize', '12, 6')
        figsize = tuple(map(int, figsize_str.split(',')))
        
        return {
            'figsize': figsize,
            'style': section.get('style', 'whitegrid'),
            'dpi': int(section.get('dpi', 300)),
            'palette': section.get('palette', 'viridis')
        }
    
    def _parse_list(self, value):
        """Parse comma-separated list of numbers."""
        return [float(x.strip()) for x in value.split(',')]
    
    def _parse_optional_int(self, value: Optional[str]):
        """Parse optional integer (handles 'null', 'None', etc.)."""
        if value is None or value.lower() in ['null', 'none', '']:
            return None
        return int(value)
    
    def get_all_params(self):
        """
        Get all configuration parameters.
        
        Returns:
            Complete configuration dictionary
        """
        return {
            'simulation': self.get_simulation_params(),
            'network': self.get_network_topology(),
            'parameters': self.get_default_parameters(),
            'functions': self.get_combination_functions(),
            'optimization': self.get_optimization_params(),
            'visualization': self.get_visualization_params()
        }


def load_config(config_path: Optional[str] = None):
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Complete configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.get_all_params()


if __name__ == "__main__":

    config = load_config()
    
    print("Configuration loaded:")
    print(f"dt: {config['simulation']['dt']}")
    print(f"num_states: {config['network']['num_states']}")
    print(f"optimization method: {config['optimization']['method']}")
