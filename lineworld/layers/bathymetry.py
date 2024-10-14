import geoalchemy2
import shapely
from core.hatching import HatchingOptions, HatchingDirection, create_hatching
from core.maptools import DocumentInfo
from geoalchemy2.shape import to_shape
from layers.elevation import ElevationLayer
from shapely.geometry import Polygon, MultiLineString, MultiPolygon
from sqlalchemy import Table, Column, Integer, Float, ForeignKey
from sqlalchemy import engine, MetaData
from sqlalchemy import select

from sqlalchemy import text
from geoalchemy2 import WKBElement

class Bathymetry(ElevationLayer):

    def __init__(self, layer_name: str, elevation_anchors: list[int | float], num_elevation_lines: int,
                 db: engine.Engine) -> None:
        super().__init__(layer_name, elevation_anchors, num_elevation_lines, db)

        metadata = MetaData()

        self.world_polygon_table = Table(
            "bathymetry_world_polygons", metadata,
            Column("id", Integer, primary_key=True),
            Column("elevation_level", Integer),
            Column("elevation_min", Float),
            Column("elevation_max", Float),
            Column("polygon", geoalchemy2.Geometry("POLYGON", srid=self.DATA_SRID.value), nullable=False)
        )

        self.map_polygon_table = Table(
            "bathymetry_map_polygons", metadata,
            Column("id", Integer, primary_key=True),
            Column("world_polygon_id", ForeignKey(f"{self.world_polygon_table.fullname}.id")),
            Column("polygon", geoalchemy2.Geometry("POLYGON", srid=self.DATA_SRID.value), nullable=False)
        )

        self.map_lines_table = Table(
            "bathymetry_map_lines", metadata,
            Column("id", Integer, primary_key=True),
            Column("map_polygon_id", ForeignKey(f"{self.map_polygon_table.fullname}.id")),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
        )

        metadata.create_all(self.db)

    def style(self, p: Polygon,
              elevation_level: int,
              document_info: DocumentInfo,
              bbox: Polygon | None = None) -> list[MultiLineString]:
        # order: ((MINX, MINY), (MINX, MAXY), (MAXX, MAXY), (MAXX, MINY), (MINX, MINY))
        # ref: https://postgis.net/docs/ST_Envelope.html
        if bbox is not None:
            bbox = [*bbox.envelope.exterior.coords[0], *bbox.envelope.exterior.coords[2]]

        # TODO debug
        # return [MultiLineString([LineString(p.exterior.coords)])]

        elevation_level_hatching_distance = [3.0 - 0.1 * i for i in range(20)]

        hatching_options = HatchingOptions()
        hatching_options.distance = elevation_level_hatching_distance[elevation_level]
        hatching_options.direction = HatchingDirection.ANGLE_135

        hatch = create_hatching(p, bbox, hatching_options)

        if hatch is not None:
            return [hatch]
        else:
            return []

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo,
            select_elevation_level: int | None = None) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        stencil = shapely.difference(document_info.get_viewport(), exclusion_zones)

        drawing_geometries = []
        with self.db.begin() as conn:
            if select_elevation_level is None:
                result = conn.execute(select(self.map_lines_table))
                drawing_geometries = [to_shape(row.lines) for row in result]
            else:
                # result = conn.execute(select(self.lines_table).where(self.lines_table.c.elevation_level == select_elevation_level))
                result = conn.execute(text(f"""
                     SELECT lines
                     FROM 
                         {self.lines_table} AS lines JOIN 
                         {self.polygon_table} AS polygons ON lines.polygon_id = polygons.id
                     WHERE 
                         polygons.elevation_level = :elevation_level

                 """), {
                    "elevation_level": select_elevation_level
                })

                drawing_geometries = [to_shape(WKBElement(row.lines)) for row in result]

        drawing_geometries_cut = []
        # remove extrusion zones
        for g in drawing_geometries:
            # drawing_geometries_cut.append(shapely.difference(g, exclusion_zones))
            drawing_geometries_cut.append(shapely.intersection(g, stencil))

        return (drawing_geometries_cut, exclusion_zones)

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
                drawing_geometries = [to_shape(row.lines) for row in result]
            else:
                # result = conn.execute(select(self.lines_table).where(self.lines_table.c.elevation_level == select_elevation_level))
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
