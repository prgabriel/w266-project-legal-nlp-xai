"""
Configuration loader with proper path resolution
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any

def get_project_root() -> Path:
    """Get the project root directory (where config.yaml is located)"""
    # Find config.yaml by traversing up the directory tree
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / 'config.yaml').exists():
            return current_dir
        current_dir = current_dir.parent
    
    # Fallback: assume script is in scripts/ subdirectory
    return Path(__file__).parent.parent

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration with proper path resolution
    
    Args:
        config_path: Optional path to config file. If None, searches for config.yaml
    
    Returns:
        Dictionary with resolved configuration
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / 'config.yaml'
    else:
        config_path = Path(config_path)
        project_root = config_path.parent
    
    # Set PROJECT_ROOT environment variable if not set
    if 'PROJECT_ROOT' not in os.environ:
        os.environ['PROJECT_ROOT'] = str(project_root)
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve paths
    if 'paths' in config:
        resolved_paths = {}
        for key, path_value in config['paths'].items():
            if isinstance(path_value, str):
                # Replace environment variables
                resolved_path = os.path.expandvars(path_value)
                
                # Convert to absolute path if relative
                if not os.path.isabs(resolved_path):
                    resolved_path = str(project_root / resolved_path)
                
                resolved_paths[key] = Path(resolved_path)
            else:
                resolved_paths[key] = path_value
        
        config['paths'] = resolved_paths
    
    # Add project root to config for reference
    config['project_root'] = project_root
    
    return config

def get_path(config: Dict[str, Any], path_key: str) -> Path:
    """
    Get a resolved path from configuration
    
    Args:
        config: Configuration dictionary
        path_key: Key for the path in config['paths']
    
    Returns:
        Resolved Path object
    """
    if 'paths' not in config or path_key not in config['paths']:
        raise KeyError(f"Path '{path_key}' not found in configuration")
    
    path = config['paths'][path_key]
    if isinstance(path, str):
        path = Path(path)
    
    return path.resolve()

# Example usage in other scripts
if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    
    print("Project Root:", config['project_root'])
    print("Data Path:", get_path(config, 'data'))
    print("Models Path:", get_path(config, 'models'))
    
    # Verify paths exist or can be created
    for path_key in ['data', 'models', 'logs']:
        path = get_path(config, path_key)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {path_key}: {path}")