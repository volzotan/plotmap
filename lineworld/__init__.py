import copy
import os
import tomllib
from typing import Any
from loguru import logger

CONFIG_FILE = "config.toml"
ENV_OVERWRITE_CONFIG = "LINEWORLD_CONFIG"

def _recursive_dict_merge(a: dict, b: dict) -> dict:
    a = copy.deepcopy(a)
    for key, value in b.items():

        if key not in a:
            a[key] = value
        elif isinstance(a[key], dict) and isinstance(value, dict):
            a[key] = _recursive_dict_merge(a[key], b[key])
        else:
            a[key] = value

    return a

def get_config() -> dict[str, Any]:
    """Load a config.toml file from disk and merge it with all LINEWORLD_* env variables
    """

    config = {}
    overwrite_config = {}

    with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)

    for name, value in os.environ.items():
        if name == ENV_OVERWRITE_CONFIG:
            logger.info(f"loading overwrite config from {value}")
            try:
                with open(value, "rb") as f:
                    overwrite_config = tomllib.load(f)
            except Exception as e:
                logger.warning(f"loading overwrite config from {value} failed: {e}")

    return _recursive_dict_merge(config, overwrite_config)

def apply_config_to_object(config: dict[str, Any], obj: Any) -> Any:
    """Replaces the value of any uppercase class variable in the given object with
    the value of the identically-named key-value pair in the config dict
    """

    altered_obj = copy.deepcopy(obj)
    members = [attr for attr in dir(altered_obj) if not callable(getattr(altered_obj, attr)) and not attr.startswith("__")]

    for key, value in config.items():
        if key.upper() in members:
            altered_obj.__setattr__(key.upper(), config[key])

    return altered_obj