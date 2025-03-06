import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import shapely
from core.maptools import DocumentInfo
import geoalchemy2
from geoalchemy2 import WKBElement
from geoalchemy2.shape import to_shape, from_shape
from shapely import STRtree, Geometry
from shapely.geometry import Polygon, MultiLineString, LineString, MultiPolygon
from shapelysmooth import taubin_smooth
from sqlalchemy import Table, Column, Integer, Float, ForeignKey
from sqlalchemy import engine, MetaData
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy import insert

from lineworld.core.maptools import Projection
from lineworld.layers.layer import Layer
from lineworld.util import gebco_grid_to_polygon
from lineworld.util.geometrytools import unpack_multipolygon, process_polygons, unpack_multilinestring

from loguru import logger


@dataclass
class ContourWorldPolygon:
    id: int | None
    elevation_level: int
    elevation_min: float
    elevation_max: float
    polygon: Polygon

    # bbox: Polygon | None = None

    def __repr__(self) -> str:
        return f"WorldPolygon [{self.id}] elevation: {self.elevation_level} | {self.elevation_min:7.2f} - {self.elevation_max:7.2f}"

    def todict(self) -> dict[str, int | float | str | WKBElement | None]:
        return {
            "elevation_level": self.elevation_level,
            "elevation_min": self.elevation_min,
            "elevation_max": self.elevation_max,
            "polygon": from_shape(self.polygon),
        }


@dataclass
class ContourMapPolygon:
    id: int | None
    world_polygon_id: int | None
    polygon: Polygon

    def __repr__(self) -> str:
        return f"MapPolygon [{self.id}]"

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "world_polygon_id": self.world_polygon_id,
            "polygon": str(from_shape(self.polygon)),  # Shapely geometry to WKB
        }


@dataclass
class ContourMapLines:
    id: int | None
    map_polygon_id: int | None
    lines: MultiLineString

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "map_polygon_id": self.map_polygon_id,
            "lines": str(from_shape(self.lines)),
        }


class Contour2(Layer):
    """
    Layer: Contour and Bathymetry (below sea-level elevation data)
    Projection Info: GEBCO data is WGS84 - Mercator (EPSG 4326)

    The Contour2 layer does not store the data from GEBCO files (WGS84) in the database
    (when transform_to_world() is invoked) but does project the GEBCO image directly
    to the map projection system. The ContourWorldPolygon dataclass is only used within the
    transform_to_map() method.

    This was necessary because performing the smoothing on the (correctly projected) DEM image
    is considerably easier than trying to do smoothing on the contour lines derived from the
    DEM in WGS84 and then converted to the map projection.

    """

    DEFAULT_DATA_URL = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2024/geotiff/"
    DATA_SRID = Projection.WGS84

    DEFAULT_LAYER_NAME = "Contour2"

    DEFAULT_GEOTIFF_SCALING_FACTOR = 0.5

    FILTER_POLYGON_MIN_AREA_MAP = 10

    DEFAULT_WINDOW_SIZE_TPI = 51
    DEFAULT_WINDOW_SIZE_SMOOTHING_LOW = 251
    DEFAULT_WINDOW_SIZE_SMOOTHING_HIGH = 501

    DEFAULT_FILTER_MIN_AREA_MAP = 1.0

    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

        self.data_dir = Path(
            Layer.DATA_DIR_NAME,
            self.config.get("layer_name", self.DEFAULT_LAYER_NAME).lower(),
        )

        self.tiles_dir = Path(self.data_dir, "tiles")
        self.scaled_dir = Path(self.data_dir, "scaled")
        self.mosaic_file = Path(self.data_dir, "gebco_mosaic.tif")
        self.projected_file = Path(self.data_dir, "projected_mosaic.tif")

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)
        if not self.tiles_dir.exists():
            os.makedirs(self.tiles_dir)
        if not self.scaled_dir.exists():
            os.makedirs(self.scaled_dir)

        metadata = MetaData()

        self.world_polygon_table = Table(
            "contour_world_polygons",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("elevation_level", Integer),
            Column("elevation_min", Float),
            Column("elevation_max", Float),
            Column(
                "polygon",
                geoalchemy2.Geography("POLYGON", srid=self.DATA_SRID.value[1]),
                nullable=False,
            ),
        )

        self.map_polygon_table = Table(
            "contour_map_polygons",
            metadata,
            Column("id", Integer, primary_key=True),
            Column(
                "world_polygon_id",
                ForeignKey(f"{self.world_polygon_table.fullname}.id"),
            ),
            Column(
                "polygon",
                geoalchemy2.Geometry("POLYGON", srid=self.DATA_SRID.value[1]),
                nullable=False,
            ),
        )

        self.map_lines_table = Table(
            "contour_map_lines",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("map_polygon_id", ForeignKey(f"{self.map_polygon_table.fullname}.id")),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False),
        )

        metadata.create_all(self.db)

    def extract(self) -> None:
        """
        Download GEBCO geoTiff files from the website and unzip them to the DATA_DIR
        """

        logger.info("extracting elevation data from GeoTiffs")

        data_url = self.config.get("data_url", self.DEFAULT_DATA_URL)

        # TODO: Download and unpack geoTiff files from the network

        # Downscaling

        dataset_files = [f for f in self.tiles_dir.iterdir() if f.is_file() and f.suffix == ".tif"]
        if len(dataset_files) == 0:
            logger.error("no GeoTiffs to transform")
            return

        scaled_files = []
        scaling_factor = self.config.get("geotiff_scaling_factor", self.DEFAULT_GEOTIFF_SCALING_FACTOR)
        if scaling_factor == 1:
            scaled_files = dataset_files
        else:
            for dataset_file in dataset_files:
                scaled_path = Path(self.scaled_dir, dataset_file.name)
                scaled_files.append(scaled_path)

                if scaled_path.exists():
                    continue

                logger.debug(f"downscaling tile: {dataset_file}")
                gebco_grid_to_polygon.downscale_and_write(dataset_file, scaled_path, scaling_factor)

        # Merging tiles into a mosaic
        if not self.mosaic_file.exists():
            logger.debug("merging mosaic tiles")
            gebco_grid_to_polygon.merge_and_write(scaled_files, self.mosaic_file)

    def transform_to_world(self) -> list[ContourWorldPolygon]:
        return []

    def transform_to_map(self, document_info: DocumentInfo) -> list[ContourMapPolygon]:
        if not self.mosaic_file.exists():
            raise Exception(f"Gebco mosaic GeoTiff {self.mosaic_file} not found")

        if not self.projected_file.exists():
            logger.debug("projecting mosaic")
            gebco_grid_to_polygon.project(self.mosaic_file, self.projected_file)

        if "elevation_anchors" not in self.config:
            logger.warning(
                f'configuration "elevation_anchors" missing for layer {self.layer_id}, fallback to default values'
            )

        worldPolygons: list[ContourWorldPolygon] = []
        layer_elevation_bounds = gebco_grid_to_polygon.get_elevation_bounds(
            self.config.get("elevation_anchors", [0, 10_000]),
            self.config.get("num_elevation_lines", 10),
        )
        logger.debug(
            f"computed elevation line bounds: {[int(layer_elevation_bounds[0][0])] + [int(x[1]) for x in layer_elevation_bounds]}"
        )

        for dataset_file in [self.projected_file]:
            logger.info(f"converting raster data to elevation contour polygons: {dataset_file})")

            with rasterio.open(dataset_file) as dataset:
                band = dataset.read(1)

                band_smoothed = gebco_grid_to_polygon._adaptive_smoothing(
                    band,
                    self.config.get("window_size_tpi", self.DEFAULT_WINDOW_SIZE_TPI),
                    self.config.get("window_size_smoothing_low", self.DEFAULT_WINDOW_SIZE_SMOOTHING_LOW),
                    self.config.get("window_size_smoothing_high", self.DEFAULT_WINDOW_SIZE_SMOOTHING_HIGH),
                )

                del band

                converted_layers = gebco_grid_to_polygon.convert(
                    dataset, band_smoothed, layer_elevation_bounds, allow_overlap=True
                )

                for i, layer in enumerate(converted_layers):
                    polys = process_polygons(
                        layer,
                        check_empty=True,
                        check_valid=True,
                        unpack=True,
                    )

                    worldPolygons += [
                        ContourWorldPolygon(
                            None,
                            i,
                            layer_elevation_bounds[i][0],
                            layer_elevation_bounds[i][1],
                            p,
                        )
                        for p in polys
                    ]

        # convert from target projection to map space
        mat = document_info.get_transformation_matrix()
        polys = [shapely.affinity.affine_transform(wp.polygon, mat) for wp in worldPolygons]

        polys = process_polygons(
            polys,
            check_valid=True,
        )

        layers: dict[int, list[ContourMapPolygon]] = {}
        for i in range(polys.shape[0]):
            geoms = np.array(unpack_multipolygon(polys[i]))

            geoms = geoms[~shapely.is_empty(geoms)]

            mask_small = shapely.area(geoms) < self.FILTER_POLYGON_MIN_AREA_MAP
            geoms = geoms[~mask_small]

            layer_number = worldPolygons[i].elevation_level
            if layer_number not in layers:
                layers[layer_number] = []

            for j in range(geoms.shape[0]):
                g = geoms[j]

                layers[layer_number].append(ContourMapPolygon(None, worldPolygons[i].id, g))

        return [x for k, v in layers.items() for x in v]

    def transform_to_lines(self, document_info: DocumentInfo) -> list[ContourMapLines]:
        with self.db.begin() as conn:
            # for some reason it's faster to not use WHERE NOT ST_IsEmpty(poly) in the SQL command

            result = conn.execute(
                text(f"""
                SELECT  mp.id, mp.polygon AS poly, ST_Envelope(mp.polygon) AS bbox
                FROM    {self.map_polygon_table} AS mp
                """)
            )

            results = result.all()
            stat: dict[str, int] = {"invalid": 0, "empty": 0, "small": 0}

            polys = [to_shape(WKBElement(x.poly)) for x in results]
            bboxes = [to_shape(WKBElement(x.bbox)) for x in results]

            stat["input"] = len(polys)

            processed_elevationPolygonLines = []
            for i, poly in enumerate(polys):
                poly = poly.segmentize(1.0)
                poly = taubin_smooth(poly, steps=self.config.get("taubin_smoothing_steps", 5))
                poly = poly.simplify(self.config.get("tolerance", 0.1))

                for g in unpack_multipolygon(poly):
                    if g.is_empty:
                        continue

                    if g.area < self.config.get("filter_min_area_map", self.DEFAULT_FILTER_MIN_AREA_MAP):
                        continue

                    for mline in self._style(g, document_info, bbox=bboxes[i]):
                        if mline.is_empty:
                            continue

                        el = ContourMapLines(None, results[i].id, mline)
                        processed_elevationPolygonLines.append(el)

        # if len(processed_elevationPolygonLines) > 0:
        #     result = conn.execute(insert(self.lines_table), [l.todict() for l in processed_elevationPolygonLines])

        stat["output"] = len(processed_elevationPolygonLines)
        logger.debug("Draw Filtering:")
        for k, v in stat.items():
            logger.debug(f"{k:10} : {v:10}")

        return processed_elevationPolygonLines

    def load(self, geometries: list[Any]) -> None:
        if geometries is None:
            return

        if len(geometries) == 0:
            logger.warning("no geometries to load. abort")
            return
        else:
            logger.info(f"loading geometries: {len(geometries)}")

        match geometries[0]:
            case ContourWorldPolygon():
                with self.db.begin() as conn:
                    conn.execute(text(f"TRUNCATE TABLE {self.world_polygon_table.fullname} CASCADE"))
                    conn.execute(
                        insert(self.world_polygon_table),
                        [g.todict() for g in geometries],
                    )

            case ContourMapPolygon():
                with self.db.begin() as conn:
                    conn.execute(text(f"TRUNCATE TABLE {self.map_polygon_table.fullname} CASCADE"))
                    conn.execute(insert(self.map_polygon_table), [g.todict() for g in geometries])

            case ContourMapLines():
                with self.db.begin() as conn:
                    conn.execute(text(f"TRUNCATE TABLE {self.map_lines_table.fullname} CASCADE"))
                    conn.execute(insert(self.map_lines_table), [g.todict() for g in geometries])

            case _:
                raise Exception(f"unknown geometry: {geometries[0]}")

    def _style(
        self,
        p: Polygon,
        document_info: DocumentInfo,
        bbox: Polygon | None = None,
    ) -> list[MultiLineString]:
        lines = [LineString(p.exterior.coords)]
        lines += [x.coords for x in p.interiors]
        return [MultiLineString(lines)]

    def out(self, exclusion_zones: list[Polygon], document_info: DocumentInfo) -> tuple[list[Geometry], list[Polygon]]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        stencil = shapely.difference(document_info.get_viewport(), shapely.unary_union(exclusion_zones))

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

            viewport_lines = shapely.intersection(stencil, np.array(drawing_geometries, dtype=MultiLineString))
            viewport_lines = viewport_lines[~shapely.is_empty(viewport_lines)]
            drawing_geometries = viewport_lines.tolist()

        # do not extend extrusion zones

        return (drawing_geometries, exclusion_zones)

    def _cut_linestring(self, ls: LineString) -> np.array:
        """
        returns NumPy array [x1, y1, x2, y2]
        """

        coordinate_pairs = np.zeros([len(ls.coords) - 1, 4], dtype=float)

        coordinate_pairs[:, 0] = ls.xy[0][:-1]
        coordinate_pairs[:, 1] = ls.xy[1][:-1]
        coordinate_pairs[:, 2] = ls.xy[0][1:]
        coordinate_pairs[:, 3] = ls.xy[1][1:]

        return coordinate_pairs

    def out_tanaka(
        self, exclusion_zones: list[Polygon], document_info: DocumentInfo, highlights: bool = False
    ) -> tuple[list[shapely.Geometry], list[Polygon]]:
        """
        correct results if (and only if) bounds are supplied in the right order, from lower to higher, ie.
        BOUNDS = get_elevation_bounds([-20, 0], LEVELS)
        """

        angle = 135
        width = 70

        output_bright = []
        output_dark = []

        # stencil = shapely.difference(document_info.get_viewport(), shapely.unary_union(exclusion_zones))

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = np.array([to_shape(row.lines) for row in result], dtype=MultiLineString)
            drawing_geometries = drawing_geometries[~shapely.is_empty(drawing_geometries)]
            drawing_geometries = unpack_multilinestring(drawing_geometries)

        # cut linestrings to single lines
        for ls in drawing_geometries:
            lines = self._cut_linestring(ls)

            # compute orientation of lines
            theta = np.degrees(np.arctan2((lines[:, 3] - lines[:, 1]), (lines[:, 2] - lines[:, 0])))

            bright_mask = (theta > (angle - width)) & (theta < (angle + width))

            for line in lines[bright_mask]:
                output_bright.append(LineString([line[:2], line[2:]]))

            for line in lines[~bright_mask]:
                output_dark.append(LineString([line[:2], line[2:]]))

        # TODO: reassemble connected lines of same color to linestrings

        # cut extrusion_zones into drawing_geometries
        # Note: using a STRtree here instead of unary_union() and difference() is a 6x speedup
        drawing_geometries_cut = []
        tree = STRtree(exclusion_zones)
        for g in output_bright if highlights else output_dark:
            g_processed = g
            for i in tree.query(g):
                g_processed = shapely.difference(g_processed, exclusion_zones[i])
                if g_processed.is_empty:
                    break
            else:
                drawing_geometries_cut.append(g_processed)

        # and do not add anything to exclusion_zones
        return (drawing_geometries_cut, exclusion_zones)


    def out_polygons(
        self,
        exclusion_zones: MultiPolygon,
        document_info: DocumentInfo,
        select_elevation_level: int | None = None,
    ) -> tuple[list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            if select_elevation_level is None:
                result = conn.execute(select(self.map_polygon_table))
                drawing_geometries = [to_shape(row.polygon) for row in result]
            else:
                result = conn.execute(
                    text(f"""
                     SELECT mp.polygon
                     FROM 
                         {self.map_polygon_table} AS mp JOIN 
                         {self.world_polygon_table} AS wp ON mp.world_polygon_id = wp.id
                     WHERE 
                         wp.elevation_level = :elevation_level
                 """),
                    {"elevation_level": select_elevation_level},
                )

                drawing_geometries = [to_shape(WKBElement(row.polygon)) for row in result]

        return (drawing_geometries, exclusion_zones)
