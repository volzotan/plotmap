from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.enums import Resampling
from rasterio.merge import merge
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform

MORPH_KERNEL_SIZE = 7


def _generate_elevation_lines(image: np.ndarray):
    """
    Generate openCV contour lines from raster images
    """

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # retrieve contours with CCOMP (only two levels of hierarchy, either hole or no hole)
    return cv2.findContours(closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)


def _get_holes_for_poly(contours, hierarchy, index):
    """
    Convert holes in openCV contour lines to shapely Points
    """
    holes = []
    next_index = hierarchy[index][0]

    while True:
        points = []

        for coord in contours[next_index]:
            points.append(coord[0])

        holes.append(points)
        next_index = hierarchy[next_index][0]

        if next_index < 0:
            break

    return holes


def get_elevation_bounds(anchors: list[int | float], num_elevation_lines: int) -> list[list[float]]:
    """
    Compute [from, to] elevation values for a total of NUM_ELEVATION_LINES.
    Example: first layer should be comprised of the terrain between 0 and 400m elevation.
    """

    layer_min_max = []
    num_splits = len(anchors) - 1

    for i in range(0, num_splits):
        elevation_line_height = (anchors[i + 1] - anchors[i]) / (num_elevation_lines / num_splits)
        for j in range(0, num_elevation_lines // num_splits):
            threshold_value_low = anchors[i] + elevation_line_height * j
            threshold_value_high = threshold_value_low + elevation_line_height
            layer_min_max.append([threshold_value_low, threshold_value_high])

    if not len(layer_min_max) == num_elevation_lines:
        raise Exception("num_elevation_lines is not conforming!")

    return layer_min_max


def downscale_and_write(input_path: Path, output_path: Path, scaling_factor: float) -> None:
    """
    Downscale GEBCO GeoTiff images
    """

    with rasterio.open(input_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * scaling_factor),
                int(src.width * scaling_factor),
            ),
            resampling=Resampling.bilinear,
        )

        transform = src.transform * src.transform.scale((src.width / data.shape[-1]), (src.height / data.shape[-2]))

        config = {
            "driver": "GTiff",
            "height": data.shape[-2],
            "width": data.shape[-1],
            "count": 1,
            "dtype": data.dtype,
            "crs": src.crs,
            "transform": transform,
        }

        with rasterio.open(output_path, "w", **config) as dst:
            dst.write(data)


def merge_and_write(geotiff_paths: list[Path], output_path: Path) -> None:
    with ExitStack() as stack:
        tiles = [stack.enter_context(rasterio.open(geotiff_path)) for geotiff_path in geotiff_paths]

        mosaic, mosaic_transform = merge(tiles, resampling=Resampling.bilinear)

        config = {
            "driver": "GTiff",
            "height": mosaic.shape[-2],
            "width": mosaic.shape[-1],
            "count": 1,
            "dtype": mosaic.dtype,
            "crs": tiles[0].crs,
            "transform": mosaic_transform,
        }

        with rasterio.open(output_path, "w", **config) as dst:
            dst.write(mosaic)


def _extract_polygons(
    band: np.ndarray,
    threshold_value_low: float,
    threshold_value_high: float,
    allow_overlap: bool,
    mask: np.ndarray | None = None,
) -> list[Polygon | MultiPolygon]:
    if mask is None:
        mask = np.zeros_like(band, dtype=np.uint8)
    else:
        mask[:, :] = 0

    if threshold_value_low > threshold_value_high:  # bathymetry
        mask[band < threshold_value_low] = 1
        if not allow_overlap:
            mask[band < threshold_value_high] = 0

    else:  # land elevation
        mask[band > threshold_value_low] = 1
        if not allow_overlap:
            mask[band > threshold_value_high] = 0

    contours, contour_hierarchy = _generate_elevation_lines(mask)

    # Debug image outpug
    # debug_im = mask.copy()
    # debug_im[debug_im > 0] = 255
    # cv2.imwrite(f"output_{threshold_value_low}_{threshold_value_high}.png", debug_im)

    if len(contours) == 0:
        return []

    contour_hierarchy = contour_hierarchy[0]
    polygons = []
    holes: dict[int, list[list[float]]] = {}

    for contour_index in range(0, len(contour_hierarchy)):
        parent_index = contour_hierarchy[contour_index][3]

        if parent_index >= 0:
            if parent_index not in holes:
                holes[parent_index] = []

            points = []
            for coord in contours[contour_index]:
                points.append(coord[0])

            holes[parent_index].append(points)

    for hierarchy_index in range(0, len(contour_hierarchy)):
        child_index = contour_hierarchy[hierarchy_index][2]
        parent_index = contour_hierarchy[hierarchy_index][3]

        if parent_index >= 0:  # is hole
            continue

        points_exterior = []
        points_interiors = []

        if child_index >= 0:  # has holes
            points_interiors = holes[hierarchy_index]
            # points_interiors = _get_holes_for_poly(contours, contour_hierarchy, child_index)
        else:  # has no holes
            pass

        for coord in contours[hierarchy_index]:
            points_exterior.append(coord[0])

        if len(points_exterior) < 3:
            logger.warning("too few points for polygon: {}".format(len(points_exterior)))
            continue

        sanitized_points_interiors = []
        for hole in points_interiors:
            if len(hole) >= 3:
                sanitized_points_interiors.append(hole)
        points_interiors = sanitized_points_interiors

        polygons.append(Polygon(points_exterior, points_interiors))

    return polygons


def convert(geotiff_path: Path, layer_min_max: list[list[float]], allow_overlap: bool = True) -> list[list[Polygon]]:
    """
    geotiff_path: path of the GeoTiff image file
    layer_min_max: min to max elevation for slicing
    allow_overlap: if true, the polygon for 2000-3000 meters will contain the polygon for 3000-4000m too. (i.e. polygons overlap like pyramids)
    """

    with rasterio.open(geotiff_path) as dataset:
        band = dataset.read(1)

        polygon_layers: list[list[Polygon]] = []
        mask = np.zeros_like(band, dtype=np.uint8)

        for layer_index in range(0, len(layer_min_max)):
            threshold_value_low = layer_min_max[layer_index][0]
            threshold_value_high = layer_min_max[layer_index][1]

            extracted_geometries = _extract_polygons(
                band,
                threshold_value_low,
                threshold_value_high,
                allow_overlap,
                mask=mask,
            )

            polygons = []
            for g in extracted_geometries:
                # convert pixel coordinates to lat lon with the geoTiff reference system
                # flip xy for openCVs row,col order
                g = transform(lambda x, y: dataset.xy(y, x), g)

                if type(g) is Polygon:
                    polygons.append(g)
                elif type(g) is MultiPolygon:
                    polygons += g.geoms
                else:
                    logger.warning("polygon is not a polygon (actual type: {})".format(type(g)))
                    continue

            polygon_layers.append(polygons)

            logger.debug(
                f"converted layer {layer_index:2d} [{threshold_value_low:9.2f} | {threshold_value_high:9.2f}] polygons: {len(polygon_layers[layer_index]):5d}"
            )

        return polygon_layers
