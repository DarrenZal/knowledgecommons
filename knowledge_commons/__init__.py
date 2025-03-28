"""
Knowledge Commons
================

A federated personal knowledge management system designed for interoperability
and knowledge sharing across communities.

This package provides tools for managing personal knowledge, creating connections
between different pieces of information, and selectively sharing with others.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

__version__ = "0.1.0"

# Global configuration object
_config = None


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the configuration for the Knowledge Commons system.
    
    Args:
        config_path: Optional path to a configuration file
        
    Returns:
        Dict containing the configuration
    """
    global _config
    
    if _config is not None and config_path is None:
        return _config
    
    # Default config location
    if config_path is None:
        config_path = os.environ.get(
            "KNOWLEDGE_COMMONS_CONFIG",
            str(Path(__file__).parent.parent / "config" / "default_config.yaml")
        )
    
    # Load the configuration
    try:
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")
    
    # Process environment variable interpolation in config
    _process_config_env_vars(_config)
    
    return _config


def _process_config_env_vars(config: Dict[str, Any]) -> None:
    """
    Process environment variable references in the configuration.
    
    Args:
        config: Configuration dictionary to process
    """
    if not isinstance(config, dict):
        return
    
    for key, value in config.items():
        if isinstance(value, dict):
            _process_config_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            if env_var in os.environ:
                config[key] = os.environ[env_var]
            # Keep the placeholder if not found - might be referring to another config entry


def _resolve_config_references(config: Dict[str, Any]) -> None:
    """
    Resolve internal references within the configuration.
    
    For example, converts "${system.data_dir}/content" to the actual path.
    
    Args:
        config: Configuration dictionary to process
    """
    # First, find all simple values that might be referenced
    refs = {}
    
    def _collect_refs(cfg, prefix=""):
        if not isinstance(cfg, dict):
            return
        
        for k, v in cfg.items():
            if isinstance(v, dict):
                _collect_refs(v, f"{prefix}{k}.")
            else:
                refs[f"{prefix}{k}"] = v
    
    _collect_refs(config)
    
    # Then, resolve references
    def _resolve_refs(cfg):
        if not isinstance(cfg, dict):
            return
        
        for k, v in cfg.items():
            if isinstance(v, dict):
                _resolve_refs(v)
            elif isinstance(v, str) and "${" in v and "}" in v:
                # Match all references in the string
                for ref_start in range(len(v)):
                    if v[ref_start:ref_start+2] == "${":
                        for ref_end in range(ref_start + 2, len(v)):
                            if v[ref_end] == "}":
                                ref_key = v[ref_start+2:ref_end]
                                if ref_key in refs:
                                    ref_value = refs[ref_key]
                                    cfg[k] = v.replace(f"${{{ref_key}}}", str(ref_value))
                                break
    
    _resolve_refs(config)


# Initialize configuration when module is imported
get_config()