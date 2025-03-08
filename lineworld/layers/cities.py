import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geoalchemy2
import numpy as np
import shapely
from core.map import DocumentInfo, Projection
from geoalchemy2.shape import from_shape, to_shape
from layers.layer import Layer
from loguru import logger
from shapely import Polygon, MultiLineString, LineString, Point
from shapely.affinity import affine_transform
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text

from lineworld.util.geometrytools import add_to_exclusion_zones
from lineworld.util.hersheyfont import HersheyFont
from lineworld.util import labelplacement


@dataclass
class CitiesLines:
    id: int | None
    circlelines: LineString
    labellines: MultiLineString

    def __repr__(self) -> str:
        return f"CitiesLines [{self.id}]"

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "circlelines": str(from_shape(self.circlelines)),
            "labellines": str(from_shape(self.labellines)),
        }


class Cities(Layer):
    DATA_URL = ""
    DATA_SRID = Projection.WGS84

    DEFAULT_LAYER_NAME = "Cities"

    DEFAULT_CITIES_FILENAME = "cities.csv"

    # simplification tolerance in WGS84 latlon, resolution: 1°=111.32km (equator worst case)
    LAT_LON_PRECISION = 0.01
    LAT_LON_MIN_SEGMENT_LENGTH = 0.1

    DEFAULT_EXCLUDE_BUFFER_DISTANCE = 2

    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

        self.data_dir = Path(
            Layer.DATA_DIR_NAME,
            self.config.get("layer_name", self.DEFAULT_LAYER_NAME).lower(),
        )
        self.cities_file = Path(
            self.data_dir,
            self.config.get("cities_filename", self.DEFAULT_CITIES_FILENAME),
        )
        self.exclude_buffer_distance = self.config.get("exclude_buffer_distance", self.DEFAULT_EXCLUDE_BUFFER_DISTANCE)

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)

        metadata = MetaData()

        self.map_lines_table = Table(
            f"{self.config_name}_cities_map_lines",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("circlelines", geoalchemy2.Geometry("LINESTRING"), nullable=False),
            Column("labellines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False),
        )

        metadata.create_all(self.db)

    def extract(self) -> None:
        pass

    def transform_to_world(self) -> None:
        pass

    def transform_to_map(self, document_info: DocumentInfo) -> None:
        pass

    def transform_to_lines(self, document_info: DocumentInfo) -> list[CitiesLines]:
        lines = []

        cities = labelplacement.read_from_file(self.cities_file, document_info, self.config)
        cities = labelplacement.generate_placement(cities, self.config)

        for c in cities:
            lines.append(
                CitiesLines(
                    None,
                    c.circle.boundary,
                    MultiLineString(c.text[c.placement]),
                )
            )

        return lines

    def load(self, geometries: list[CitiesLines]) -> None:
        if geometries is None:
            return

        if len(geometries) == 0:
            logger.warning("no geometries to load. abort")
            return
        else:
            logger.info(f"loading geometries: {len(geometries)}")

        with self.db.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {self.map_lines_table.fullname} CASCADE"))
            conn.execute(insert(self.map_lines_table), [g.todict() for g in geometries])

    def _out(
        self,
        column_name: str,
        exclusion_zones: list[Polygon],
        document_info: DocumentInfo,
    ) -> tuple[list[shapely.Geometry], list[Polygon]]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        stencil = shapely.difference(document_info.get_viewport(), shapely.unary_union(exclusion_zones))

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row._asdict()[column_name]) for row in result]

            viewport_lines = shapely.intersection(stencil, np.array(drawing_geometries, dtype=MultiLineString))
            viewport_lines = viewport_lines[~shapely.is_empty(viewport_lines)]
            drawing_geometries = viewport_lines.tolist()

        # and add buffered lines to exclusion_zones
        exclusion_zones = add_to_exclusion_zones(
            drawing_geometries,
            exclusion_zones,
            self.exclude_buffer_distance,
            self.config.get("tolerance_exclusion_zones", 0.5),
        )

        return (drawing_geometries, exclusion_zones)


class CitiesLabels(Cities):
    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

    def out(
        self, exclusion_zones: list[Polygon], document_info: DocumentInfo
    ) -> tuple[list[shapely.Geometry], list[Polygon]]:
        return self._out("labellines", exclusion_zones, document_info)


class CitiesCircles(Cities):
    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

    def out(
        self, exclusion_zones: list[Polygon], document_info: DocumentInfo
    ) -> tuple[list[shapely.Geometry], list[Polygon]]:
        return self._out("circlelines", exclusion_zones, document_info)
