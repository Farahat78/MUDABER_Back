"""
Configuration Loader
Loads and validates YAML configuration
"""

import os
import yaml
from typing import Dict, Any


class Config:
    """Configuration manager"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = None
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_all(self) -> Dict:
        """Get entire configuration"""
        return self._config

    def validate(self) -> bool:
        """Validate required configuration keys"""
        required_keys = [
            'data.base_dir',
            'data.unified_dataset',
            'training.models',
            'prediction.strategy'
        ]

        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Missing required config key: {key}")

        return True


def get_config(config_path: str = None) -> Config:
    """Factory function to get config instance"""
    if config_path is None:
        config_path = "config.yaml"

    return Config(config_path)
