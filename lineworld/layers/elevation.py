import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.maptools import DocumentInfo, Projection
from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from layers.layer import Layer
from shapely import to_wkt
from shapely.affinity import affine_transform, translate
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import text
from util import gebco_grid_to_polygon

from lineworld.util.geometrytools import *


@dataclass
class ElevationWorldPolygon():
    id: int | None
    elevation_level: int
    elevation_min: float
    elevation_max: float
    polygon: Polygon

    # bbox: Polygon | None = None

    def __repr__(self) -> str:
        return (
            f"WorldPolygon [{self.id}] elevation: {self.elevation_level} | {self.elevation_min:7.2f} - {self.elevation_max:7.2f}")

    def todict(self) -> dict[str, int | float | str | WKBElement | None]:
        return {
            "elevation_level": self.elevation_level,
            "elevation_min": self.elevation_min,
            "elevation_max": self.elevation_max,
            "polygon": from_shape(self.polygon)
        }


@dataclass
class ElevationMapPolygon():
    id: int | None
    world_polygon_id: int | None
    polygon: Polygon

    def __repr__(self) -> str:
        return (
            f"MapPolygon [{self.id}]")

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "world_polygon_id": self.world_polygon_id,
            "polygon": str(from_shape(self.polygon))  # Shapely geometry to WKB
        }


@dataclass
class ElevationMapLines():
    id: int | None
    map_polygon_id: int | None
    lines: MultiLineString

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "map_polygon_id": self.map_polygon_id,
            "lines": str(from_shape(self.lines))
        }


class ElevationLayer(Layer):
    """
    Layer: Contour and Bathymetry (below sea-level elevation data)
    Projection Info: GEBCO data is WGS84 - Mercator (EPSG 4326)
    """

    DATA_URL = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2024/geotiff/"
    DATA_SRID = Projection.WGS84

    LAYER_NAME = "Elevation"
    DATA_DIR = Path("data", LAYER_NAME.lower())
    TILES_DIR = Path(DATA_DIR, "tiles")
    SCALED_DIR = Path(DATA_DIR, "scaled")
    MOSAIC_FILE = Path(DATA_DIR, "gebco_mosaic.tif")

    GEOTIFF_SCALING_FACTOR = 0.5

    # minimal area of polygons on a WGS84 geoid in m^2
    FILTER_POLYGON_MIN_AREA_WGS84 = 1e4

    # minimal area of polygons on the map (in map units, mm^2)
    FILTER_POLYGON_MIN_AREA_MAP = 1.0

    # simplification tolerance in WGS84 latlon, resolution: 1°=111.32km (equator worst case)
    LAT_LON_PRECISION = 0.01

    LAT_LON_MIN_SEGMENT_LENGTH = 0.1

    WRAPOVER_LONGITUDE_EXTENSION = 60

    def __init__(self, layer_name: str, elevation_anchors: list[int | float], num_elevation_lines: int,
                 db: engine.Engine) -> None:
        super().__init__(layer_name, db)

        self.ELEVATION_ANCHORS = elevation_anchors
        self.NUM_ELEVATION_LINES = num_elevation_lines

        if not self.DATA_DIR.exists():
            os.makedirs(self.DATA_DIR)
        if not self.TILES_DIR.exists():
            os.makedirs(self.TILES_DIR)
        if not self.SCALED_DIR.exists():
            os.makedirs(self.SCALED_DIR)

    def extract(self) -> None:
        """
        Download GEBCO geoTiff files from the website and unzip them to the DATA_DIR
        """

        logger.info("extracting elevation data from GeoTiffs")

        # Downscaling
        dataset_files = [f for f in self.TILES_DIR.iterdir() if f.is_file() and f.suffix == ".tif"]
        if len(dataset_files) == 0:
            logger.warning("no GeoTiffs to transform")

        scaled_files = []
        for dataset_file in dataset_files:
            scaled_path = Path(self.SCALED_DIR, dataset_file.name)
            scaled_files.append(scaled_path)

            if scaled_path.exists():
                continue

            logger.debug(f"downscaling tile: {dataset_file}")
            gebco_grid_to_polygon.downscale_and_write(dataset_file, scaled_path, self.GEOTIFF_SCALING_FACTOR)

        # Merging tiles into a mosaic
        if not self.MOSAIC_FILE.exists():
            logger.debug("merging mosaic tiles")
            gebco_grid_to_polygon.merge_and_write(scaled_files, self.MOSAIC_FILE)

    def transform(self) -> list[ElevationWorldPolygon]:
        """
        Transform GEBCO geoTiff raster image data to shapely polygons on layers with fixed min-max elevations
        """

        if not self.MOSAIC_FILE.exists():
            raise Exception(f"Gebco mosaic GeoTiff {self.MOSAIC_FILE} not found")

        polygons: list[ElevationWorldPolygon] = []
        layer_elevation_bounds = gebco_grid_to_polygon.get_elevation_bounds(
            self.ELEVATION_ANCHORS,
            self.NUM_ELEVATION_LINES
        )
        logger.debug(
            f"computed elevation line bounds: {[int(layer_elevation_bounds[0][0])] + [int(x[1]) for x in layer_elevation_bounds]}")

        for dataset_file in [self.MOSAIC_FILE]:
            logger.info(f"converting raster data to elevation contour polygons: {dataset_file})")

            converted_layers = gebco_grid_to_polygon.convert(
                dataset_file,
                layer_elevation_bounds,
                allow_overlap=True
            )

            for layer_index in range(len(converted_layers)):
                polys = process_polygons(
                    converted_layers[layer_index],
                    simplify_precision=self.LAT_LON_PRECISION,
                    min_area_wgs84=self.FILTER_POLYGON_MIN_AREA_WGS84,
                    check_empty=True,
                    check_valid=True,
                    unpack=True
                )

                polygons += [
                    ElevationWorldPolygon(
                        None,
                        layer_index,
                        layer_elevation_bounds[layer_index][0],
                        layer_elevation_bounds[layer_index][1],
                        polys[i]
                    ) for i in range(polys.shape[0])
                ]

        return polygons

    def load(self, geometries: list[Any]) -> None:

        if geometries is None:
            return

        if len(geometries) == 0:
            logger.warning("no geometries to load. abort")
            return
        else:
            logger.info(f"loading geometries: {len(geometries)}")

        match geometries[0]:

            case ElevationWorldPolygon():
                with self.db.begin() as conn:
                    result = conn.execute(text(f"TRUNCATE TABLE {self.world_polygon_table.fullname} CASCADE"))
                    result = conn.execute(insert(self.world_polygon_table), [g.todict() for g in geometries])

            case ElevationMapPolygon():
                with self.db.begin() as conn:
                    result = conn.execute(text(f"TRUNCATE TABLE {self.map_polygon_table.fullname} CASCADE"))
                    result = conn.execute(insert(self.map_polygon_table), [g.todict() for g in geometries])

            case ElevationMapLines():
                with self.db.begin() as conn:
                    result = conn.execute(text(f"TRUNCATE TABLE {self.map_lines_table.fullname} CASCADE"))
                    result = conn.execute(insert(self.map_lines_table), [g.todict() for g in geometries])

            case _:
                raise Exception(f"unknown geometry: {geometries[0]}")

    def project(self, document_info: DocumentInfo) -> list[ElevationMapPolygon]:

        with self.db.begin() as conn:

            params = {
                "srid": document_info.projection.value[1],
                "min_area": self.FILTER_POLYGON_MIN_AREA_WGS84
            }

            result_center = conn.execute(text(f"""
                SELECT  id,
                        elevation_level,
                        polygon AS poly
                FROM {self.world_polygon_table.fullname}
                WHERE ST_Area(polygon) >= :min_area
            """), params)

            results = result_center.all()

            polygons = [to_shape(WKBElement(x.poly)) for x in results]

            if document_info.wrapover:
                select_slice = text(f"""
                    SELECT  id,
                            elevation_level,
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
                results = results + result_right + result_left

            polys = np.array(polygons, dtype=Polygon)

            logger.debug(f"project: loaded polygons: {polys.shape[0]}")

            if self.LAT_LON_MIN_SEGMENT_LENGTH is not None:
                polys = shapely.segmentize(polys, self.LAT_LON_MIN_SEGMENT_LENGTH)

            # from WGS to target projection
            func = document_info.get_projection_func(self.DATA_SRID)
            polys = polys.tolist()
            polys = [shapely.ops.transform(func, p) for p in polys]

            # from target projection to map space
            mat = document_info.get_transformation_matrix()
            polys = [affine_transform(p, mat) for p in polys]

            polys = process_polygons(
                polys,
                simplify_precision=document_info.tolerance,
                check_valid=True
            )

            layers: dict[int, list[ElevationMapPolygon]] = {}
            for i in range(polys.shape[0]):
                geoms = np.array(unpack_multipolygon(polys[i]))

                # geoms = geoms[~shapely.is_empty(geoms)]

                mask_small = shapely.area(geoms) < self.FILTER_POLYGON_MIN_AREA_MAP
                geoms = geoms[~mask_small]

                for j in range(geoms.shape[0]):
                    g = geoms[j]

                    layer_number = results[i].elevation_level
                    if layer_number not in layers:
                        layers[layer_number] = []

                    layers[layer_number].append(ElevationMapPolygon(None, results[i].id, g))

            return self._cut_layers(layers)
            # return [x for k, v in layers.items() for x in v]

    def _cut_layers(self, layers: dict[int, list[ElevationMapPolygon]]) -> list[ElevationMapPolygon]:

        num_layers = max(layers.keys()) + 1

        result_list = []

        cutting_polygon = MultiPolygon()
        for i in reversed(range(num_layers)):
            current_layer_polygon = shapely.unary_union(
                np.array([p.polygon for p in layers[i]], dtype=Polygon))
            cutting_polygon_new = shapely.union(cutting_polygon, current_layer_polygon)

            for emp in layers[i]:
                new_geometry = shapely.difference(emp.polygon, cutting_polygon)
                for g in unpack_multipolygon(new_geometry):
                    if g.is_empty:
                        continue

                    result_list.append(ElevationMapPolygon(None, emp.world_polygon_id, g))

            cutting_polygon = cutting_polygon_new

        return result_list

    def draw(self, document_info: DocumentInfo) -> list[ElevationMapLines]:
        with self.db.begin() as conn:

            # for some reason it's faster to not use WHERE NOT ST_IsEmpty(poly) in the SQL command

            result = conn.execute(text(f"""
                SELECT  mp.id, wp.elevation_level, mp.polygon AS poly, ST_Envelope(mp.polygon) AS bbox
                FROM {self.map_polygon_table} AS mp JOIN 
                        {self.world_polygon_table} AS wp ON mp.world_polygon_id = wp.id
                """))

            results = result.all()
            stat: dict[str, int] = {
                "invalid": 0,
                "empty": 0,
                "small": 0
            }

            polys = np.array([to_shape(WKBElement(x.poly)) for x in results], dtype=Polygon)
            bboxes = [to_shape(WKBElement(x.bbox)) for x in results]

            stat["input"] = polys.shape[0]

            processed_elevationPolygonLines = []
            for i in range(polys.shape[0]):
                for mline in self.style(polys[i], results[i].elevation_level, document_info, bbox=bboxes[i]):

                    if mline.is_empty:
                        continue

                    el = ElevationMapLines(None, results[i].id, mline)
                    processed_elevationPolygonLines.append(el)

        # if len(processed_elevationPolygonLines) > 0:
        #     result = conn.execute(insert(self.lines_table), [l.todict() for l in processed_elevationPolygonLines])

        stat["output"] = len(processed_elevationPolygonLines)
        logger.debug("Convert Filtering:")
        for k, v in stat.items():
            logger.debug(f"{k:10} : {v:10}")

        return processed_elevationPolygonLines

    def style(self, p: Polygon, document_info: DocumentInfo, bbox: Polygon | None = None) -> list[MultiLineString]:
        raise NotImplementedError("Must override method")
