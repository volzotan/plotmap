import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fiona
import geoalchemy2
import numpy as np
import shapely
from core.maptools import DocumentInfo, Projection
from geoalchemy2.shape import from_shape, to_shape
from layers.layer import Layer
from loguru import logger
from shapely import Polygon, MultiLineString, MultiPolygon, LineString, Point
from shapely.affinity import affine_transform
from shapely.geometry import shape
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text

from lineworld.util.geometrytools import add_to_exclusion_zones
from lineworld.util.hersheyfont import HersheyFont


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

    DEFAULT_CITIES_FILENAME = "world-townspots-z5.json"
    DEFAULT_LABELS_FILENAME = "world-labels-z5.json"

    # simplification tolerance in WGS84 latlon, resolution: 1°=111.32km (equator worst case)
    LAT_LON_PRECISION = 0.01
    LAT_LON_MIN_SEGMENT_LENGTH = 0.1

    DEFAULT_FONT_SIZE = 5
    DEFAULT_CITY_CIRCLE_RADIUS = 2

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
        self.labels_file = Path(
            self.data_dir,
            self.config.get("labels_filename", self.DEFAULT_LABELS_FILENAME),
        )
        self.font_size = self.config.get("font_size", self.DEFAULT_FONT_SIZE)
        self.city_circle_radius = self.config.get("city_circle_radius", self.DEFAULT_CITY_CIRCLE_RADIUS)
        self.exclude_buffer_distance = self.config.get("exclude_buffer_distance", self.DEFAULT_EXCLUDE_BUFFER_DISTANCE)

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)

        metadata = MetaData()

        self.map_lines_table = Table(
            "cities_map_lines",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("circlelines", geoalchemy2.Geometry("LINESTRING"), nullable=False),
            Column("labellines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False),
        )

        metadata.create_all(self.db)

        self.font = HersheyFont()

    def extract(self) -> None:
        pass

    def transform_to_world(self) -> None:
        pass

    def transform_to_map(self, document_info: DocumentInfo) -> None:
        pass

    def transform_to_lines(self, document_info: DocumentInfo) -> list[CitiesLines]:
        project_func = document_info.get_projection_func(self.DATA_SRID)
        mat = document_info.get_transformation_matrix()

        city_name = []
        city_pos = []
        city_label = []

        lines = []

        for item in fiona.open(self.cities_file):
            shapefile_geom = shape(item["geometry"])
            geom = shapely.ops.transform(project_func, shapefile_geom)
            geom = affine_transform(geom, mat)

            if type(geom) is not Point:
                raise Exception(f"parsing shapefile: unexpected type: {geom}")

            # for key in item["properties"]:
            #     print("{} : {}".format(key, item["properties"][key]))
            # exit()

            city_pos.append(geom)
            city_name.append(item["properties"]["name"])  # label placement computation is done with name, not asciiname
            # city_name.append(item["properties"]["asciiname"])

        for item in fiona.open(self.labels_file):
            shapefile_geom = shape(item["geometry"])
            geom = shapely.ops.transform(project_func, shapefile_geom)
            geom = affine_transform(geom, mat)

            if type(geom) is not Polygon:
                raise Exception(f"parsing shapefile: unexpected type: {geom}")

            city_label.append(geom)

        for i in range(0, len(city_name)):
            minx, _, _, maxy = city_label[i].bounds
            c = [minx, maxy]
            text_lines = MultiLineString(self.font.lines_for_text(city_name[i], self.font_size))
            text_lines = shapely.affinity.translate(
                text_lines, xoff=c[0] + self.city_circle_radius - 0.75, yoff=c[1] + 0.4
            )
            lines.append(
                CitiesLines(
                    None,
                    city_pos[i].buffer(self.city_circle_radius).exterior,
                    text_lines,
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
    ) -> tuple[list[shapely.Geometry], MultiPolygon]:
        return self._out("labellines", exclusion_zones, document_info)


class CitiesCircles(Cities):
    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

    def out(
        self, exclusion_zones: list[Polygon], document_info: DocumentInfo
    ) -> tuple[list[shapely.Geometry], MultiPolygon]:
        return self._out("circlelines", exclusion_zones, document_info)
