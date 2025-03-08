import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geoalchemy2
import shapely
from lineworld.core.map import DocumentInfo, Projection
from geoalchemy2.shape import from_shape, to_shape
from lineworld.layers.layer import Layer
from loguru import logger
from shapely import Polygon, MultiLineString
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, String, Integer
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text

from lineworld.util.hersheyfont import HersheyFont


@dataclass
class MetaLines:
    id: int | None
    text: str
    lines: MultiLineString

    def __repr__(self) -> str:
        return f"MetaLines [{self.id}]: {self.text}"

    def todict(self) -> dict[str, int | float | str | None]:
        return {"text": self.text, "lines": str(from_shape(self.lines))}


class Meta(Layer):
    DATA_URL = ""
    DATA_SRID = Projection.WGS84

    DEFAULT_LAYER_NAME = "Meta"

    DEFAULT_FONT_SIZE = 12

    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

        self.data_dir = Path(
            Layer.DATA_DIR_NAME,
            self.config.get("layer_name", self.DEFAULT_LAYER_NAME).lower(),
        )

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)

        metadata = MetaData()

        self.map_lines_table = Table(
            f"{self.config_name}_meta_map_lines",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("text", String, nullable=False),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False),
        )

        metadata.create_all(self.db)

        self.font = HersheyFont()

    def extract(self) -> None:
        pass

    def transform_to_world(self) -> None:
        pass

    def transform_to_map(self, document_info: DocumentInfo) -> None:
        pass

    def transform_to_lines(self, document_info: DocumentInfo) -> list[MetaLines]:
        padding = document_info.get_viewport_padding()

        metaLines = []

        metaLines.append(
            MetaLines(
                None,
                "THE WORLD",
                shapely.affinity.translate(
                    MultiLineString(self.font.lines_for_text("THE WORLD", 18)),
                    xoff=padding[3],
                    yoff=document_info.height - 4,
                ),
            )
        )

        scale = (40_075 * 1000 * 100) / document_info.width

        metaLines.append(
            MetaLines(
                None,
                None,
                shapely.affinity.translate(
                    MultiLineString(self.font.lines_for_text(f"Scale: 1:{int(scale)}", 9)),
                    xoff=padding[3] + 120,
                    yoff=document_info.height - 4,
                ),
            )
        )

        return metaLines

    def load(self, geometries: list[MetaLines]) -> None:
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

    def out(
        self, exclusion_zones: list[Polygon], document_info: DocumentInfo
    ) -> tuple[list[shapely.Geometry], list[Polygon]]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

        return (drawing_geometries, exclusion_zones)
