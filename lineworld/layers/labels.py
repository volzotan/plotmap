import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fiona
import geoalchemy2
import numpy as np
import shapely
from HersheyFonts import HersheyFonts
from core.maptools import DocumentInfo, Projection
from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from layers.layer import Layer
from loguru import logger
from shapely import to_wkt, Polygon, MultiLineString, MultiPolygon, LineString, Point
from shapely.affinity import affine_transform, translate
from shapely.geometry import shape
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, String, Integer, ForeignKey
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text

from lineworld.core.hatching import HatchingDirection, HatchingOptions, create_hatching
from lineworld.util import downloader
from lineworld.util.geometrytools import process_polygons, unpack_multipolygon, hershey_text_to_lines, \
    add_to_exclusion_zones


@dataclass
class LabelsLines():
    id: int | None
    text: str
    lines: MultiLineString

    def __repr__(self) -> str:
        return (
            f"LabelsLines [{self.id}]: {self.text}")

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "text": self.text,
            "lines": str(from_shape(self.lines))
        }

class Labels(Layer):

    DATA_URL = ""
    DATA_SRID = Projection.WGS84

    DEFAULT_LAYER_NAME = "Labels"
    DEFAULT_LABELS_FILENAME = "labels.json"

    # simplification tolerance in WGS84 latlon, resolution: 1°=111.32km (equator worst case)
    LAT_LON_PRECISION = 0.01
    LAT_LON_MIN_SEGMENT_LENGTH = 0.1

    DEFAULT_EXCLUDE_BUFFER_DISTANCE = 2
    DEFAULT_FONT_SIZE = 12

    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

        self.data_dir = Path(Layer.DATA_DIR_NAME, self.config.get("layer_name", self.DEFAULT_LAYER_NAME).lower())
        self.labels_file = Path(self.data_dir, self.config.get("labels_filename", self.DEFAULT_LABELS_FILENAME))
        self.font_size = self.config.get("font_size", self.DEFAULT_FONT_SIZE)

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)

        metadata = MetaData()

        self.map_lines_table = Table("labels_map_lines", metadata,
            Column("id", Integer, primary_key=True),
            Column("text", String, nullable=False),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
        )

        metadata.create_all(self.db)

        self.hfont = HersheyFonts()
        self.hfont.load_default_font("futural")
        self.hfont.normalize_rendering(self.font_size)

    def extract(self) -> None:
        pass

    def transform_to_world(self) -> None:
        pass

    def transform_to_map(self, document_info: DocumentInfo) -> None:
        pass

    def transform_to_lines(self, document_info: DocumentInfo) -> list[LabelsLines]:
        if not self.labels_file.exists():
            logger.warning(f"labels file {self.labels_file} not found")
            return []

        project_func = document_info.get_projection_func(self.DATA_SRID)
        mat = document_info.get_transformation_matrix()

        labellines = []

        with open(self.labels_file) as f:
            data = json.load(f)

            for label_data in data["labels"]:
                pos = shapely.ops.transform(project_func, Point(reversed(label_data[0])))
                pos = affine_transform(pos, mat)

                sub_labels = label_data[1].split("\n")

                for i, sub_label in enumerate(sub_labels):
                    lines = hershey_text_to_lines(self.hfont, sub_label)

                    center_offset = shapely.envelope(lines).centroid

                    mat_font = document_info.get_transformation_matrix_font(
                        xoff=pos.x - center_offset.x,
                        yoff=pos.y - center_offset.y + (self.font_size * 1.06) * i
                    )

                    labellines.append(LabelsLines(None, sub_label, affine_transform(lines, mat_font)))

        return labellines

    def load(self, geometries: list[LabelsLines]) -> None:

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


    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        stencil = shapely.difference(document_info.get_viewport(), exclusion_zones)

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

            viewport_lines = shapely.intersection(stencil, np.array(drawing_geometries, dtype=MultiLineString))
            viewport_lines = viewport_lines[~shapely.is_empty(viewport_lines)]
            drawing_geometries = viewport_lines.tolist()

        # and add buffered lines to exclusion_zones
        exclusion_zones = add_to_exclusion_zones(
            drawing_geometries, exclusion_zones,
            self.config.get("exclude_buffer_distance", self.DEFAULT_EXCLUDE_BUFFER_DISTANCE),
            self.config.get("tolerance", 0.1))

        return (drawing_geometries, exclusion_zones)