import numpy as np
import shapely
from HersheyFonts import HersheyFonts
from loguru import logger
from pyproj import Geod
from shapely import Geometry
from shapely.geometry import GeometryCollection, Polygon, MultiPolygon, LineString, MultiLineString


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


def unpack_multipolygon(g: Geometry | list[Geometry] | np.ndarray) -> list[Polygon]:
    return _unpack_multigeometry(g, Polygon)


def unpack_multilinestring(g: Geometry | list[Geometry] | np.ndarray) -> list[LineString]:
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
        min_area_mm2: float | None = None) -> np.array:
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


def hershey_text_to_lines(font: HersheyFonts, text: str) -> MultiLineString:
    lines_raw = font.lines_for_text(text)
    # lines_restructured = []
    # for (x1, y1), (x2, y2) in lines_raw:
    #     lines_restructured.append([[x1, y1], [x2, y2]])
    # lines = MultiLineString(lines_restructured)

    return MultiLineString([[[x1, y1], [x2, y2]] for (x1, y1), (x2, y2) in lines_raw])


def add_to_exclusion_zones(drawing_geometries: list[Geometry], exclusion_zones: MultiPolygon, exclude_buffer: float,
                           simplification_tolerance: float = 0.1) -> MultiPolygon:
    cutting_tool = shapely.unary_union(np.array(drawing_geometries))
    cutting_tool = cutting_tool.buffer(exclude_buffer)
    cutting_tool = shapely.simplify(cutting_tool, simplification_tolerance)
    return shapely.union(exclusion_zones, cutting_tool)
