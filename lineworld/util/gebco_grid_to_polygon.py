from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject, calculate_default_transform
from shapely import LineString
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform
from shapelysmooth import taubin_smooth, chaikin_smooth

from lineworld.core.svgwriter import SvgWriter
from lineworld.util.geometrytools import unpack_multipolygon

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


def project(source_path: Path, destination_path: Path, destination_projection: str = "ESRI:54029") -> None:
    with rasterio.open(source_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, destination_projection, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": destination_projection, "transform": transform, "width": width, "height": height})

        with rasterio.open(destination_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                band_arr = src.read(i)

                reproject(
                    source=band_arr,
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=destination_projection,
                    resampling=Resampling.nearest,
                )


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


def convert(
    dataset: rasterio.DatasetBase, band: np.ndarray, layer_min_max: list[list[float]], allow_overlap: bool = True
) -> list[list[Polygon]]:
    """
    geotiff_path: path of the GeoTiff image file
    layer_min_max: min to max elevation for slicing
    allow_overlap: if true, the polygon for 2000-3000 meters will contain the polygon for 3000-4000m too. (i.e. polygons overlap like pyramids)
    """

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


def _calculate_topographic_position_index(data: np.ndarray, window_size: int) -> np.ndarray:
    """Simplified version of the Topographic Position Index
    See: "Weiss, A., 2001. Topographic Position and Landforms Analysis"
    """

    if window_size % 2 != 1:
        logger.warning("window size is not an odd number, resulting TPI will be skewed")

    data_positive = (data - np.min(data)).astype(float)
    kernel = np.ones([window_size, window_size], dtype=float)
    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 0
    tpi = data_positive - (cv2.filter2D(data_positive, -1, kernel) / np.sum(kernel))
    return tpi


def _adaptive_smoothing(
    data: np.ndarray, window_size_tpi: int, window_size_smoothing_low: int, window_size_smooting_high: int
) -> np.ndarray:
    """Adaptive Smoothing for DEM data
    Based on the paper "A design of contour generation for topographic maps with adaptive DEM smoothing"
    by Kettunen et al. https://www.tandfonline.com/doi/full/10.1080/23729333.2017.1300998
    """

    tpi = _calculate_topographic_position_index(data, window_size_tpi)
    data_smoothed_low = cv2.blur(data, (window_size_smoothing_low, window_size_smoothing_low)).astype(np.int16)
    data_smoothed_high = cv2.blur(data, (window_size_smooting_high, window_size_smooting_high)).astype(np.int16)
    normalized_tpi = np.abs(tpi) / np.abs(np.max(tpi))
    data_smoothed = (normalized_tpi * data_smoothed_high + (1 - normalized_tpi) * data_smoothed_low).astype(np.int16)

    return data_smoothed


def _debug_save_image(
    output_path: Path,
    data: np.ndarray,
    resize_dimensions: tuple[int, int] | None = None,
    normalize_max: float = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if resize_dimensions is not None:
        data = cv2.resize(data, resize_dimensions)

    if normalize_max is None:
        normalize_max = np.max(np.abs(data))

    # normalize around 0
    data = data / normalize_max

    # center around 0.5 (colormap range [0,1])
    data = (data / 2) + 0.5

    cmap = mpl.colormaps["seismic"]
    color_mapped_data = cmap(data)
    plt.imsave(str(output_path), color_mapped_data, format="png")


if __name__ == "__main__":
    INPUT_FILE = Path("experiments/hatching/data/GebcoToBlender/fullsize_reproject.tif")
    INPUT_FILE = Path("experiments/hatching/data/gebco_crop.tif")

    OUTPUT_FILE = Path("output.svg")

    layer_elevation_bounds = get_elevation_bounds([0, 10_000], 20)

    with rasterio.open(INPUT_FILE) as dataset:
        band = dataset.read(1)

        WINDOW_SIZE_TPI = 51
        WINDOW_SIZE_SMOOTHING_LOW = 251
        WINDOW_SIZE_SMOOTHING_HIGH = 501

        # tpi = _calculate_topographic_position_index(band, WINDOW_SIZE_TPI)
        # band_smoothed_low = cv2.blur(band, (WINDOW_SIZE_SMOOTHING_LOW, WINDOW_SIZE_SMOOTHING_LOW))
        # band_smoothed_high = cv2.blur(band, (WINDOW_SIZE_SMOOTHING_HIGH, WINDOW_SIZE_SMOOTHING_HIGH))

        # _debug_save_image("tpi.png", tpi, (5000, 5000))
        # _debug_save_image("low.png", band_smoothed_low, (5000, 5000), normalize_max=10_000)
        # _debug_save_image("high.png", band_smoothed_high, (5000, 5000), normalize_max=10_000)

        # normalized_tpi = np.abs(tpi) / np.abs(np.max(tpi))
        # band_smoothed = normalized_tpi * band_smoothed_high + (1 - normalized_tpi) * band_smoothed_low
        # _debug_save_image("comb.png", band_smoothed, (5000, 5000), normalize_max=10_000)
        # _debug_save_image("orig.png", band, (5000, 5000), normalize_max=10_000)

        converted_layers = convert(dataset, band, layer_elevation_bounds, allow_overlap=True)

        lines = []
        for layer in converted_layers:
            for p in layer:
                lines += [LineString(p.exterior.coords)]
                lines += [LineString(x.coords) for x in p.interiors]

        svg = SvgWriter(OUTPUT_FILE, band.shape)

        options = {"fill": "none", "stroke": "black", "stroke-width": "1"}
        svg.add("original", lines, options=options)

        band_smoothed = _adaptive_smoothing(
            band, WINDOW_SIZE_TPI, WINDOW_SIZE_SMOOTHING_LOW, WINDOW_SIZE_SMOOTHING_HIGH
        )

        options_blurred = {**options, "stroke": "red"}
        converted_layers = convert(dataset, band_smoothed, layer_elevation_bounds, allow_overlap=True)
        lines_blurred = []
        for layer in converted_layers:
            for p in layer:
                lines_blurred += [LineString(p.exterior.coords)]
                lines_blurred += [LineString(x.coords) for x in p.interiors]

        svg.add("blurred", lines_blurred, options=options_blurred)

        SEGMENTIZE_VALUE = 10
        SIMPLIFY_VALUE = 0.5

        options_segmentized = {**options, "stroke": "yellow"}
        lines_blurred_segmentized = []
        for l in lines_blurred:
            processed = l.segmentize(SEGMENTIZE_VALUE)
            if type(processed) == LineString and not processed.is_empty:
                lines_blurred_segmentized.append(processed)
        svg.add("segmentized", lines_blurred_segmentized, options=options_segmentized)

        options_smoothed_t = {**options, "stroke": "green"}
        svg.add(
            "smoothed_t", [taubin_smooth(l, steps=100) for l in lines_blurred_segmentized], options=options_smoothed_t
        )

        svg.write()
