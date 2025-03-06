import numpy as np
import shapely
from loguru import logger
from pyproj import Geod
from shapely import Geometry
from shapely.geometry import (
    GeometryCollection,
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
)


def _unpack_multigeometry[T](g: Geometry | list[Geometry] | np.ndarray, geometry_type: T) -> list[T]:
    single: Geometry = None
    multi: Geometry = None

    if geometry_type == Polygon:
        single, multi = (Polygon, MultiPolygon)
    elif geometry_type == LineString:
        single, multi = (LineString, MultiLineString)
    else:
        raise Exception(f"unknown geometry_type for _unpack_multigeometry: {geometry_type}")

    unpacked: list[T] = []
    packed = None
    if type(g) is np.ndarray:
        packed = g.tolist()
    elif type(g) is list:
        packed = g
    else:
        packed = [g]

    for e in packed:
        match e:
            case None:
                continue
            case single():
                unpacked.append(e)
            case multi():
                unpacked += e.geoms
            case GeometryCollection():
                for gc in e.geoms:
                    unpacked += _unpack_multigeometry(gc, geometry_type)
            case _:
                # logger.warning(f"ignoring geometry: {type(e)}")
                pass

    return unpacked


def crop_geometry(main: list[Geometry] | Geometry, tool: list[Geometry]) -> list[Geometry] | Geometry:
    # TODO
    pass


def _linestring_to_coordinate_pairs(
    linestring: LineString,
) -> list[list[tuple[float, float]]]:
    pairs = []

    for i in range(len(linestring.coords) - 1):
        pairs.append([linestring.coords[i], linestring.coords[i + 1]])

    return pairs


def unpack_multipolygon(g: Geometry | list[Geometry] | np.ndarray) -> list[Polygon]:
    return _unpack_multigeometry(g, Polygon)


def unpack_multilinestring(
    g: Geometry | list[Geometry] | np.ndarray,
) -> list[LineString]:
    return _unpack_multigeometry(g, LineString)


def calculate_geodesic_area(p: Polygon) -> float:
    poly_area, _ = Geod(ellps="WGS84").geometry_area_perimeter(p)
    return poly_area


def process_polygons(
    polygons: list[Polygon],
    simplify_precision: float | None = None,
    check_valid: bool = False,
    unpack: bool = False,
    check_empty: bool = False,
    min_area_wgs84: float | None = None,
    min_area_mm2: float | None = None,
) -> np.array:
    stat: dict[str, int] = {
        "input": 0,
        "output": 0,
    }

    if len(polygons) == 0:
        return np.array([], dtype=Polygon)

    polys = np.array(polygons, dtype=Polygon)

    stat["input"] = polys.shape[0]

    if simplify_precision is not None:
        polys = shapely.simplify(polys, simplify_precision)

    if check_valid:
        stat["invalid"] = 0
        stat["invalid"] = np.count_nonzero(~shapely.is_valid(polys))
        polys = shapely.make_valid(polys)

    if unpack:
        polys = np.array(unpack_multipolygon(polys))

    if check_empty:
        stat["empty"] = 0
        mask_empty = shapely.is_empty(polys)
        stat["empty"] += np.count_nonzero(mask_empty)
        polys = polys[~mask_empty]

    if min_area_wgs84 is not None:
        stat["small"] = 0
        mask_small = np.vectorize(lambda p: calculate_geodesic_area(p) < min_area_wgs84)(polys)
        stat["small"] += np.count_nonzero(mask_small)
        polys = polys[~mask_small]

    if min_area_mm2 is not None:
        stat["small"] = 0
        mask_small = shapely.area(polys) < min_area_mm2
        stat["small"] += np.count_nonzero(mask_small)
        polys = polys[~mask_small]

    stat["output"] = polys.shape[0]

    logger.debug("Filtering:")
    for k, v in stat.items():
        logger.debug(f"{k:10} : {v:10}")

    return polys


def add_to_exclusion_zones(
    drawing_geometries: list[Geometry],
    exclusion_zones: list[Polygon],
    exclude_buffer: float,
    simplification_tolerance: float = 0.5,
) -> list[Polygon]:
    # Note for .buffer(): reducing the quad segments from 8 (default) to 4 gives a speedup of ~40%

    new_zones = shapely.simplify(drawing_geometries, simplification_tolerance)
    new_zones = [shapely.buffer(g, exclude_buffer, quad_segs=4) for g in new_zones]
    return new_zones + exclusion_zones
