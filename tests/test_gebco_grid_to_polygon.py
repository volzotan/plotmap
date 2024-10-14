import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from lineworld.util import gebco_grid_to_polygon


@pytest.fixture
def single_poly() -> np.ndarray:
    band = np.zeros((1000, 2000, 1), np.uint8)
    cv2.rectangle(band, (100, 100), (1900, 900), (100), -1)
    cv2.rectangle(band, (200, 200), (1800, 800), (200), -1)
    cv2.imwrite("test_single_poly.png", band)
    return band


@pytest.mark.parametrize("allow_overlap", [True, False])
def test_convert_single_poly(single_poly: np.ndarray, allow_overlap: bool) -> None:
    results = gebco_grid_to_polygon._extract_polygons(single_poly, 50, 150, allow_overlap)

    assert len(results) == 1
    assert type(results[0]) is Polygon

    if allow_overlap:
        assert len(results[0].interiors) == 0  # no hole
    else:
        assert len(results[0].interiors) == 1  # has hole


@pytest.fixture
def poly_with_hole() -> np.ndarray:
    band = np.zeros((1000, 2000, 1), np.uint8)
    cv2.rectangle(band, (100, 100), (1900, 900), (100), -1)
    cv2.rectangle(band, (200, 200), (1800, 800), (200), -1)
    cv2.rectangle(band, (300, 300), (1700, 700), (0), -1)
    cv2.imwrite("test_poly_with_hole.png", band)
    return band


@pytest.mark.parametrize("allow_overlap", [True, False])
def test_convert_poly_with_hole(poly_with_hole: np.ndarray, allow_overlap: bool) -> None:
    results = gebco_grid_to_polygon._extract_polygons(poly_with_hole, 50, 100, allow_overlap)

    assert len(results) == 1
    assert type(results[0]) is Polygon

    holes = results[0].interiors
    assert len(holes) == 1


@pytest.fixture
def poly_with_hole_and_island() -> np.ndarray:
    band = np.zeros((1000, 2000, 1), np.uint8)
    cv2.rectangle(band, (100, 100), (1900, 900), (100), -1)
    cv2.rectangle(band, (200, 200), (1800, 800), (200), -1)
    cv2.rectangle(band, (300, 300), (1700, 700), (0), -1)
    cv2.rectangle(band, (400, 400), (1600, 600), (250), -1)
    cv2.imwrite("test_poly_with_hole_and_island.png", band)
    return band


@pytest.mark.parametrize("allow_overlap", [True, False])
def test_convert_poly_with_hole(poly_with_hole_and_island: np.ndarray, allow_overlap: bool) -> None:
    results = gebco_grid_to_polygon._extract_polygons(poly_with_hole_and_island, 50, 255, allow_overlap)

    assert len(results) == 2
    for geom in results:
        assert type(geom) is Polygon

    if len(results[0].interiors) == 0:
        results = list(reversed(results))

    assert len(results[0].interiors) == 1
    assert len(results[1].interiors) == 0


@pytest.fixture
def poly_with_multiple_holes() -> np.ndarray:
    band = np.zeros((1000, 2000, 1), np.uint8)
    cv2.rectangle(band, (100, 100), (1900, 900), (100), -1)
    cv2.rectangle(band, (200, 200), (300, 300), (0), -1)
    cv2.rectangle(band, (200, 400), (300, 500), (0), -1)
    cv2.rectangle(band, (200, 600), (300, 700), (0), -1)
    cv2.imwrite("test_poly_with_multiple_holes.png", band)
    return band


@pytest.mark.parametrize("allow_overlap", [True, False])
def test_convert_poly_with_multiple_holes(poly_with_multiple_holes: np.ndarray, allow_overlap: bool) -> None:
    results = gebco_grid_to_polygon._extract_polygons(poly_with_multiple_holes, 50, 255, allow_overlap)

    assert len(results) == 1
    for geom in results:
        assert type(geom) is Polygon

    assert len(results[0].interiors) == 3



@pytest.fixture
def poly_with_multiple_layers() -> np.ndarray:
    band = np.zeros((1000, 2000, 1), np.uint8)
    cv2.rectangle(band, (100, 100), (900, 900), (100), -1)
    cv2.rectangle(band, (200, 200), (800, 800), (150), -1)
    cv2.rectangle(band, (300, 300), (700, 700), (200), -1)
    cv2.rectangle(band, (400, 400), (600, 600), (250), -1)
    cv2.imwrite("poly_with_multiple_layers.png", band)
    return band


def approx_90(actual_value: float, expected_value: float) -> bool:
    return abs((actual_value / expected_value)) - 1 < 0.1

@pytest.mark.parametrize("allow_overlap", [True, False])
def test_convert_poly_with_multiple_layers(poly_with_multiple_layers: np.ndarray, allow_overlap: bool) -> None:

    mask = np.zeros_like(poly_with_multiple_layers, dtype=np.uint8)

    results0 = gebco_grid_to_polygon._extract_polygons(poly_with_multiple_layers, 50, 125, allow_overlap)
    results1 = gebco_grid_to_polygon._extract_polygons(poly_with_multiple_layers, 125, 175, allow_overlap, mask=mask)
    results2 = gebco_grid_to_polygon._extract_polygons(poly_with_multiple_layers, 175, 225, allow_overlap, mask=mask)
    results3 = gebco_grid_to_polygon._extract_polygons(poly_with_multiple_layers, 225, 255, allow_overlap, mask=mask)

    if allow_overlap:
        assert approx_90(results0[0].area, 800**2)
        assert approx_90(results1[0].area, 600**2)
        assert approx_90(results2[0].area, 400**2)
        assert approx_90(results3[0].area, 200**2)
    else:
        assert approx_90(results0[0].area, 800**2 - 600**2)
        assert approx_90(results1[0].area, 600**2 - 400**2)
        assert approx_90(results2[0].area, 400**2 - 200**2)
        assert approx_90(results3[0].area, 200**2)

