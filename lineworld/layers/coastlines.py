import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fiona
import geoalchemy2
import numpy as np
import shapely
from core.maptools import DocumentInfo, Projection
from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from layers.layer import Layer
from loguru import logger
from shapely import to_wkt, Polygon, MultiLineString, MultiPolygon, LineString
from shapely.affinity import affine_transform, translate
from shapely.geometry import shape
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text

from lineworld.core.hatching import HatchingDirection, HatchingOptions, create_hatching
from lineworld.util import downloader
from lineworld.util.geometrytools import process_polygons, unpack_multipolygon


@dataclass
class LandPolygon():
    id: int | None
    polygon: Polygon

    def __repr__(self) -> str:
        return (
            f"LandPolygon [{self.id}]")

    def todict(self) -> dict[str, int | float | str | WKBElement | None]:
        return {
            "polygon": from_shape(self.polygon)  # Shapely geometry to WKB
        }


@dataclass
class CoastlineLines():
    id: int | None
    polygon_id: int | None
    lines: MultiLineString

    def __repr__(self) -> str:
        return (
            f"CoastlineLines [{self.id}]")

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "polygon_id": self.polygon_id,
            "lines": str(from_shape(self.lines))
        }


class Coastlines(Layer):
    DEFAULT_DATA_URL = "https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip"
    DATA_SRID = Projection.WGS84

    DEFAULT_LAYER_NAME = "Coastlines"

    # minimal area of land polygons on the map (in map units, mm^2)
    FILTER_POLYGON_MIN_AREA_MAP = 1.0
    FILTER_POLYGON_MIN_AREA_WGS84 = 3e7

    # simplification tolerance in WGS84 latlon, resolution: 1°=111.32km (equator worst case)
    LAT_LON_PRECISION = 0.01
    LAT_LON_MIN_SEGMENT_LENGTH = 0.1
    WRAPOVER_LONGITUDE_EXTENSION = 60

    DEFAULT_BUFFER_DISTANCE = 2
    DEFAULT_HATCHING_DISTANCE = 2.0

    def __init__(self, layer_id: str, db: engine.Engine, config: [str, Any]) -> None:
        super().__init__(layer_id, db, config)

        self.data_url = self.config.get("data_url", self.DEFAULT_DATA_URL)

        self.data_dir = Path(Layer.DATA_DIR_NAME, self.config.get("layer_name", self.DEFAULT_LAYER_NAME).lower())
        self.shapefile_dir = Path(self.data_dir, Path(self.data_url).stem)

        self.buffer_distance = self.config.get("buffer_distance", self.DEFAULT_BUFFER_DISTANCE)
        self.hatching_distance = self.config.get("hatching_distance", self.DEFAULT_HATCHING_DISTANCE)

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)
        if not self.shapefile_dir.exists():
            os.makedirs(self.shapefile_dir)

        metadata = MetaData()

        self.world_polygon_table = Table("coastlines_world_polygons", metadata,
                                         Column("id", Integer, primary_key=True),
                                         Column("polygon",
                                                geoalchemy2.Geography("POLYGON", srid=self.DATA_SRID.value[1]),
                                                nullable=False)
                                         )

        self.map_lines_table = Table("coastlines_map_lines", metadata,
                                     Column("id", Integer, primary_key=True),
                                     Column("polygon_id", ForeignKey(f"{self.world_polygon_table.fullname}.id")),
                                     Column("lines",
                                            geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
                                     )

        metadata.create_all(self.db)

    def extract(self) -> None:

        filename_zip = Path(self.data_dir, Path(self.DATA_URL).name)

        if not filename_zip.exists():
            logger.info(f"Downloading: {self.data_url}")
            downloader.download_file(self.data_url, filename_zip)

        shapefiles = [f for f in self.shapefile_dir.iterdir() if f.is_file() and f.suffix == ".shp"]
        if len(shapefiles) == 0:
            logger.info(f"Unpacking OSM land polygon shapefiles: {filename_zip.name}")
            shutil.unpack_archive(filename_zip, self.data_dir)

    def transform_to_world(self) -> list[LandPolygon]:

        logger.info("extracting land polygons from OSM shapefile")

        shapefiles = [Path(f) for f in self.shapefile_dir.iterdir() if f.is_file() and f.suffix == ".shp"]

        if len(shapefiles) == 0:
            logger.warning("no shapefiles to transform")

        geometries: list[Polygon] = []
        for shapefile in shapefiles:
            with fiona.open(shapefile) as f:
                geometries += [shape(item["geometry"]) for item in f]

        logger.debug("reading shapefiles completed")

        polys = process_polygons(
            geometries,
            simplify_precision=self.LAT_LON_PRECISION,
            check_empty=True,
            check_valid=True,
            unpack=True
        )

        return [LandPolygon(None, polys[i]) for i in range(polys.shape[0])]

    def transform_to_map(self, document_info: DocumentInfo) -> None:
        pass

    def transform_to_lines(self, document_info: DocumentInfo) -> list[CoastlineLines]:
        with self.db.begin() as conn:
            params = {
                "srid": document_info.projection.value[1],
                "min_area": self.FILTER_POLYGON_MIN_AREA_WGS84
            }

            # result = conn.execute(text(f"""
            #     SELECT  id,
            #             ST_Transform(
            #                 polygon ::geometry,
            #                 :srid
            #             ) AS poly
            #     FROM {self.world_polygon_table.fullname}
            #     WHERE ST_Area(polygon) >= :min_area
            # """), params)

            # result = conn.execute(text(f"""
            #     SELECT  ST_Union(
            #                 ST_MakeValid(
            #                     ST_Transform(polygon ::geometry, :srid)
            #                 )
            #             ) AS poly
            #     FROM {self.world_polygon_table.fullname}
            #     WHERE ST_Area(polygon) >= :min_area
            # """), params)

            result_center = conn.execute(text(f"""
                SELECT  id,
                        polygon AS poly
                FROM {self.world_polygon_table.fullname}
                WHERE ST_Area(polygon) >= :min_area
            """), params)

            results = result_center.all()
            mat = document_info.get_transformation_matrix()

            polygons = [to_shape(WKBElement(x.poly)) for x in results]

            if document_info.wrapover:
                select_slice = text(f"""
                    SELECT  id,
                            ST_MakeValid(ST_Intersection(
                                polygon,
                                ST_GeogFromText(:viewport)
                            ) ::geometry) AS poly
                    FROM {self.world_polygon_table.fullname}
                    WHERE ST_Area(polygon) >= :min_area
                """)

                result_right = conn.execute(
                    select_slice,
                    {**params, "viewport":
                        to_wkt(shapely.box(-180, 85, -180 + self.WRAPOVER_LONGITUDE_EXTENSION, -85))
                     }
                ).all()

                result_left = conn.execute(
                    select_slice,
                    {**params, "viewport":
                        to_wkt(shapely.box(180 - self.WRAPOVER_LONGITUDE_EXTENSION, 85, 180, -85))}
                ).all()

                polygons_right = [translate(to_shape(WKBElement(x.poly)), xoff=360) for x in result_right]
                polygons_left = [translate(to_shape(WKBElement(x.poly)), xoff=-360) for x in result_left]

                polygons = polygons + polygons_left + polygons_right

            logger.debug(f"loaded {len(polygons)} polygons for conversion")

            polygons = shapely.segmentize(np.array(polygons), self.LAT_LON_MIN_SEGMENT_LENGTH)

            polygons = [shapely.ops.transform(document_info.get_projection_func(self.DATA_SRID), p) for p in polygons]
            polygons = [affine_transform(p, mat) for p in polygons]

            polygons = process_polygons(
                polygons,
                simplify_precision=self.config.get("tolerance", 0.1) * 2,
                min_area_mm2=self.FILTER_POLYGON_MIN_AREA_MAP,
                check_empty=True,
                check_valid=True,
                unpack=True
            )

            mlines = self._style(polygons, document_info)

            processed_coastlinesPolygonLines = [CoastlineLines(None, None, mline) for mline in mlines]

            return processed_coastlinesPolygonLines

    def _style(self, polygons: np.ndarray, document_info: DocumentInfo) -> list[
        MultiLineString]:

        inner = shapely.unary_union(polygons)

        logger.debug("union done")

        buffered = inner.buffer(self.buffer_distance)
        buffered = shapely.difference(buffered, inner)

        logger.debug("buffer + difference done")

        coastlines = []
        for p in unpack_multipolygon(inner):
            coastlines.append(LineString(p.exterior.coords))
            coastlines += [LineString(hole.coords) for hole in p.interiors]

        logger.debug("hatching begin")

        hatching_options = HatchingOptions()
        hatching_options.distance = self.hatching_distance
        hatching_options.direction = HatchingDirection.ANGLE_45

        hatch = create_hatching(buffered, [0, 0, document_info.width, document_info.height], hatching_options)

        if hatch is not None:
            return [MultiLineString(coastlines), hatch]
        else:
            return [MultiLineString(coastlines)]

    def load(self, geometries: list[LandPolygon | CoastlineLines]) -> None:

        if geometries is None:
            return

        if len(geometries) == 0:
            logger.warning("no geometries to load. abort")
            return
        else:
            logger.info(f"loading geometries: {len(geometries)}")

        match geometries[0]:
            case LandPolygon():
                with self.db.begin() as conn:
                    conn.execute(text(f"TRUNCATE TABLE {self.world_polygon_table.fullname} CASCADE"))
                    conn.execute(insert(self.world_polygon_table), [g.todict() for g in geometries])

            case CoastlineLines():
                with self.db.begin() as conn:
                    conn.execute(text(f"TRUNCATE TABLE {self.map_lines_table.fullname} CASCADE"))
                    conn.execute(insert(self.map_lines_table), [g.todict() for g in geometries])

            case _:
                raise Exception(f"unknown geometry: {geometries[0]}")

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

    def project2(self, document_info: DocumentInfo) -> list[CoastlineLines]:
        with self.db.begin() as conn:
            params = {
                "srid": document_info.projection.value[1],
                "min_area": self.FILTER_POLYGON_MIN_AREA_WGS84
            }

            result = conn.execute(text("""
                WITH polys AS (
                    SELECT  id,
                            ST_MakeValid(
                                ST_Simplify(
                                    ST_Transform(
                                        polygon::geometry,
                                        3857
                                    ), 
                                    10000
                                )
                            ) AS poly
                    FROM coastlines_world_polygons
                    WHERE ST_Area(polygon) >= 1000000000
                ), unions AS (
                    SELECT ST_Union(poly) AS un
                    FROM polys
                )
                SELECT ST_Difference(ST_Buffer(un, 200000), un) as buffered from unions
            """), params)

            results = result.all()
            mat = document_info.get_transformation_matrix()

            logger.debug(f"loaded {len(results)} polygons for conversion")

            polygons = [to_shape(WKBElement(x.buffered)) for x in results]
            polygons = [affine_transform(p, mat) for p in polygons]

            mlines = self.style2(polygons, document_info)
            processed_coastlinesPolygonLines = [CoastlineLines(None, None, mline) for mline in mlines]

            return processed_coastlinesPolygonLines

    def style2(self, polygons: np.ndarray, document_info: DocumentInfo) -> list[
        MultiLineString]:

        logger.debug("style")

        buffered = shapely.unary_union(polygons)

        logger.debug("union done")

        hatching_options = HatchingOptions()
        hatching_options.distance = 2.0
        hatching_options.direction = HatchingDirection.ANGLE_45

        hatch = create_hatching(buffered, [0, 0, document_info.width, document_info.height], hatching_options)

        if hatch is not None:
            return [MultiLineString(hatch)]
        else:
            return [MultiLineString()]
