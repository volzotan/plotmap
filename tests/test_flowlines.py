import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from scipy import ndimage
from shapely import Point

from lineworld.core.flowlines import FlowlineHatcherConfig, FlowlineTiler, FlowlineTilerPoly
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.export import convert_svg_to_png
from lineworld.util.rastertools import normalize_to_uint8
from lineworld.util.slope import get_slope

@pytest.fixture
def resize_size() -> tuple[float | int]:
    return (500, 500)

@pytest.fixture
def elevation(resize_size: tuple[float | int]) -> np.ndarray:
    ELEVATION_FILE = Path("experiments/hatching/data/gebco_crop.tif")

    data = None
    with rasterio.open(str(ELEVATION_FILE)) as dataset:
        data = dataset.read()[0]

    data = cv2.resize(data, resize_size)

    return data

@pytest.fixture
def flow_config() -> FlowlineHatcherConfig:
    config = FlowlineHatcherConfig()
    return config

@pytest.fixture
def mapping(elevation: np.ndarray, output_path: Path, flow_config: FlowlineHatcherConfig) -> list[np.ndarray]:

    elevation[elevation > 0] = 0  # bathymetry data only

    _, _, _, _, angles, inclination = get_slope(elevation, 1)

    mapping_angle = angles  # float

    mapping_non_flat = np.zeros_like(inclination, dtype=np.uint8)
    mapping_non_flat[inclination > flow_config.MIN_INCLINATION] = 255  # uint8

    mapping_distance = normalize_to_uint8(elevation)  # uint8

    mapping_max_segments = np.full_like(angles, int(255/2))

    mapping_angle = cv2.blur(mapping_angle, (10, 10))
    mapping_distance = cv2.blur(mapping_distance, (10, 10))
    mapping_max_segments = cv2.blur(mapping_max_segments, (10, 10))

    cv2.imwrite(str(Path(output_path, "mapping_angle.png")), normalize_to_uint8(mapping_angle / math.tau))
    cv2.imwrite(str(Path(output_path, "mapping_non_flat.png")), mapping_non_flat.astype(np.uint8) * 255)
    cv2.imwrite(str(Path(output_path, "mapping_distance.png")), mapping_distance)
    cv2.imwrite(str(Path(output_path, "mapping_max_segments.png")), mapping_max_segments)

    return [mapping_angle, mapping_non_flat, mapping_distance, mapping_max_segments]


def test_flowlines_tiler_square(
        mapping: list[np.ndarray],
        output_path: Path,
        resize_size: tuple[float | int],
        flow_config: FlowlineHatcherConfig):

    flow_config.COLLISION_APPROXIMATE = True

    tiler = FlowlineTiler(mapping, flow_config, (2, 2))
    linestrings = tiler.hatch()

    svg_path = Path(output_path, "test_flowlines_tiler_square.svg")
    svg = SvgWriter(svg_path, resize_size)
    options = {"fill": "none", "stroke": "black", "stroke-width": "1"}
    svg.add("flowlines", linestrings, options=options)
    svg.write()

    convert_svg_to_png(svg_path, svg.dimensions[0] * 10)


def test_flowlines_tiler_poly(
        mapping: list[np.ndarray],
        output_path: Path,
        resize_size: tuple[float | int],
        flow_config: FlowlineHatcherConfig):

    flow_config.COLLISION_APPROXIMATE = True

    tiler = FlowlineTilerPoly(
        mapping,
        flow_config,
        [Point([resize_size[0]//2, resize_size[0]//2]).buffer(min(resize_size)*0.49)]
    )
    linestrings = tiler.hatch()

    svg_path = Path(output_path, "test_flowlines_tiler_poly.svg")
    svg = SvgWriter(svg_path, resize_size)
    options = {"fill": "none", "stroke": "black", "stroke-width": "1"}
    svg.add("flowlines", linestrings, options=options)
    svg.write()

    convert_svg_to_png(svg_path, svg.dimensions[0] * 10)
