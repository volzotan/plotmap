from typing import Any

from sqlalchemy import engine
from loguru import logger


class Layer:
    DATA_DIR_NAME = "data"

    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]):
        self.layer_id = layer_id
        self.db = db

        if "layer" not in config or layer_id not in config["layer"]:
            logger.warning(f"layer {layer_id} has no configuration entry. Using default configuration.")

        self.config = config.get("layer", {}).get(layer_id, {})

    def transform_to_world(self):
        pass

    def transform_to_map(self):
        pass

    def transform_to_lines(self):
        pass

    def load(self):
        pass

    def out(self):
        pass
