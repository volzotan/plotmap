from typing import Any

import numpy as np
import shapely
from core.maptools import DocumentInfo
import geoalchemy2
from geoalchemy2 import WKBElement
from geoalchemy2.shape import to_shape
from layers.elevation import ElevationLayer
from shapely.geometry import Polygon, MultiLineString, LineString, MultiPolygon
from shapely.geometry.polygon import InteriorRingSequence
from sqlalchemy import Table, Column, Integer, Float, ForeignKey
from sqlalchemy import engine, MetaData
from sqlalchemy import select
from sqlalchemy import text

from lineworld.layers.elevation import ElevationMapPolygon


class Contour(ElevationLayer):

    def __init__(self,
                 layer_name: str,
                 db: engine.Engine,
                 config: dict[str, Any]) -> None:
        super().__init__(layer_name, db, config)

        self.config = {**self.config, **config.get("layer", {}).get("contour", {})}

        metadata = MetaData()

        self.world_polygon_table = Table(
            "contour_world_polygons", metadata,
            Column("id", Integer, primary_key=True),
            Column("elevation_level", Integer),
            Column("elevation_min", Float),
            Column("elevation_max", Float),
            Column("polygon", geoalchemy2.Geography("POLYGON", srid=self.DATA_SRID.value[1]), nullable=False)
        )

        self.map_polygon_table = Table(
            "contour_map_polygons", metadata,
            Column("id", Integer, primary_key=True),
            Column("world_polygon_id", ForeignKey(f"{self.world_polygon_table.fullname}.id")),
            Column("polygon", geoalchemy2.Geometry("POLYGON", srid=self.DATA_SRID.value[1]), nullable=False)
        )

        self.map_lines_table = Table(
            "contour_map_lines", metadata,
             Column("id", Integer, primary_key=True),
            Column("map_polygon_id", ForeignKey(f"{self.map_polygon_table.fullname}.id")),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
        )

        metadata.create_all(self.db)

    def transform_to_map(self, document_info: DocumentInfo) -> list[ElevationMapPolygon]:
        return super().transform_to_map(document_info, allow_overlap=True)

    def _style(self, p: Polygon, elevation_level: int,
              document_info: DocumentInfo, bbox: Polygon | None = None) -> list[MultiLineString]:

        lines = [LineString(p.exterior.coords)]
        lines += [x.coords for x in p.interiors]
        return [MultiLineString(lines)]

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

        # do not extend extrusion zones

        return (drawing_geometries, exclusion_zones)



    def out_polygons(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo,
                     select_elevation_level: int | None = None) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            if select_elevation_level is None:
                result = conn.execute(select(self.map_polygon_table))
                drawing_geometries = [to_shape(row.polygon) for row in result]
            else:
                result = conn.execute(text(f"""
                     SELECT mp.polygon
                     FROM 
                         {self.map_polygon_table} AS mp JOIN 
                         {self.world_polygon_table} AS wp ON mp.world_polygon_id = wp.id
                     WHERE 
                         wp.elevation_level = :elevation_level
                 """), {
                    "elevation_level": select_elevation_level
                })

                drawing_geometries = [to_shape(WKBElement(row.polygon)) for row in result]

        return (drawing_geometries, exclusion_zones)
