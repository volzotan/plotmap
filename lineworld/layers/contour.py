import shapely
from core.maptools import DocumentInfo
import geoalchemy2
from geoalchemy2.shape import to_shape
from layers.elevation import ElevationLayer
from shapely.geometry import Polygon, MultiLineString, LineString, MultiPolygon
from sqlalchemy import Table, Column, Integer, Float, ForeignKey
from sqlalchemy import engine, MetaData
from sqlalchemy import select

from lineworld.layers.elevation import ElevationMapPolygon


class Contour(ElevationLayer):

    def __init__(self, layer_name: str, elevation_anchors: list[int | float], num_elevation_lines: int,
                 db: engine.Engine) -> None:
        super().__init__(layer_name, elevation_anchors, num_elevation_lines, db)

        metadata = MetaData()

        self.world_polygon_table = Table("contour_world_polygons", metadata,
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

        self.map_lines_table = Table("contour_map_lines", metadata,
             Column("id", Integer, primary_key=True),
             Column("map_polygon_id", ForeignKey(f"{self.map_polygon_table.fullname}.id")),
             Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
        )

        metadata.create_all(self.db)

    def project(self, document_info: DocumentInfo) -> list[ElevationMapPolygon]:
        return super().project(document_info, allow_overlap=True)

    def style(self, p: Polygon, elevation_level: int,
              document_info: DocumentInfo, bbox: Polygon | None = None) -> list[MultiLineString]:
        return [MultiLineString([LineString(p.exterior.coords)])]

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

        return (drawing_geometries, exclusion_zones)
