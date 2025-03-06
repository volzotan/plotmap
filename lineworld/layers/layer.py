from typing import Any

from shapely import Polygon, Geometry
from sqlalchemy import engine
from loguru import logger

from lineworld.core.maptools import DocumentInfo


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

    def transform_to_map(self, document_info: DocumentInfo) -> list[Any]:
        pass

    def transform_to_lines(self, document_info: DocumentInfo) -> list[Any]:
        pass

    def load(self, geometries: list[Any]) -> None:
        pass

    def out(self, exclusion_zones: list[Polygon], document_info: DocumentInfo) -> tuple[list[Geometry], list[Polygon]]:
        pass
