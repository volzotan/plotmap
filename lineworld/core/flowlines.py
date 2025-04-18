import concurrent
import datetime
import itertools
import math

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import cv2
import numpy as np
import rasterio
import shapely
from loguru import logger
from scipy import ndimage
from shapely import LineString, Polygon, Point, MultiLineString

from lineworld.core.svgwriter import SvgWriter
from lineworld.util import geometrytools
from lineworld.util.export import convert_svg_to_png
from lineworld.util.geometrytools import crop_linestrings
from lineworld.util.rastertools import normalize_to_uint8
from lineworld.util.slope import get_slope


import flowlines_py

MAX_ITERATIONS = 20_000_000


class Mapping(StrEnum):
    DISTANCE = (auto(),)
    ANGLE = (auto(),)
    MAX_LENGTH = (auto(),)
    FLAT = (auto(),)


@dataclass
class FlowlineHatcherConfig:
    # distance between lines in mm
    LINE_DISTANCE: tuple[float, float] | list[float] = (0.3, 5.0)
    LINE_DISTANCE_END_FACTOR = 0.5

    # distance between points constituting a line in mm
    LINE_STEP_DISTANCE: float = 0.3

    LINE_MAX_LENGTH: tuple[int, int] = (10, 50)

    # max difference (in radians) in slope between line points
    MAX_ANGLE_DISCONTINUITY: float = math.pi / 2
    MIN_INCLINATION: float = 0.001  # 50.0

    # How many line segments should be skipped before the next seedpoint is extracted
    SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: int = 5

    # SCALE_ADJUSTMENT_VALUE: float = 0.3

    COLLISION_APPROXIMATE: bool = True
    VIZ_LINE_THICKNESS: int = 5


def _py_config_to_rust_config(
    pyc: FlowlineHatcherConfig, rsc: flowlines_py.FlowlinesConfig
) -> flowlines_py.FlowlinesConfig:
    rsc.line_distance = pyc.LINE_DISTANCE
    rsc.line_distance_end_factor = pyc.LINE_DISTANCE_END_FACTOR
    rsc.line_step_distance = pyc.LINE_STEP_DISTANCE
    rsc.line_max_length = pyc.LINE_MAX_LENGTH
    rsc.max_angle_discontinuity = pyc.MAX_ANGLE_DISCONTINUITY
    # rsc.min_inclincation = pyc.MIN_INCLINATION
    rsc.starting_point_init_distance = [pyc.LINE_DISTANCE[0] * 1.5, pyc.LINE_DISTANCE[0] * 1.5]
    rsc.seedpoint_extraction_skip_line_segments = pyc.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS
    rsc.max_iterations = MAX_ITERATIONS

    return rsc


def _compute_rust(
    p: Polygon, bbox: list[float], tile_mappings: list[np.ndarray], config: FlowlineHatcherConfig
) -> list[LineString]:
    min_col, min_row, max_col, max_row = bbox

    minx, miny, maxx, maxy = p.bounds
    dimensions = (int(maxx - minx), int(maxy - miny))

    rust_config = flowlines_py.FlowlinesConfig()
    rust_config = _py_config_to_rust_config(config, rust_config)

    rust_lines: list[list[tuple[float, float]]] = flowlines_py.hatch(dimensions, rust_config, *tile_mappings)
    linestrings = [shapely.affinity.translate(LineString(l), xoff=minx, yoff=miny) for l in rust_lines]

    return crop_linestrings(linestrings, p)


def _compute_python(
    p: Polygon, bbox: list[float], tile_mappings: list[np.ndarray], config: FlowlineHatcherConfig
) -> list[LineString]:
    min_col, min_row, max_col, max_row = bbox
    dimensions = (max_col - min_col, max_row - min_row)

    hatcher = FlowlineHatcher(dimensions, tile_mappings, config)
    linestrings = hatcher.hatch()
    linestrings = [shapely.affinity.translate(ls, xoff=min_col, yoff=min_row) for ls in linestrings]

    return crop_linestrings(linestrings, p)


class FlowlineTilerPoly:
    def __init__(
        self,
        mappings: dict[Mapping, np.ndarray],
        config: FlowlineHatcherConfig,
        polygons_map: list[Polygon],
        polygons_raster: list[Polygon],
        use_rust: bool = False,
    ):
        self.mappings = mappings
        self.config = config
        self.polygons_map = polygons_map
        self.polygons_raster = polygons_raster
        self.use_rust = use_rust

        self.tiles = [{"linestrings": []} for _ in polygons_map]

    def hatch(self) -> list[LineString]:
        all_linestrings = []

        with ProcessPoolExecutor() as executor:
            futures = []

            for i in range(len(self.polygons_map)):
                p_m = self.polygons_map[i]
                p_r = self.polygons_raster[i]
                logger.debug(f"processing tile {i:03}/{len(self.polygons_map):03} : {i / len(self.polygons_map) * 100.0:5.2f}%")

                min_col, min_row, max_col, max_row = [int(e) for e in shapely.bounds(p_r).tolist()]
                max_col += 1
                max_row += 1

                if max_row - min_row < 3 or max_col - min_col < 3:
                    logger.warning(f"empty tile {p_m}")
                    continue

                tile_mappings = [
                    self.mappings[Mapping.DISTANCE],
                    self.mappings[Mapping.ANGLE],
                    self.mappings[Mapping.MAX_LENGTH],
                    self.mappings[Mapping.FLAT],
                ]
                tile_mappings = [mapping[min_row:max_row, min_col:max_col] for mapping in tile_mappings]

                bbox = [min_col, min_row, max_col, max_row]

                if self.use_rust:
                    futures.append(executor.submit(_compute_rust, p_m, bbox, tile_mappings, self.config))
                else:
                    futures.append(executor.submit(_compute_python, p_m, bbox, tile_mappings, self.config))

            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            for future in done:
                all_linestrings += future.result()

        return all_linestrings


class FlowlineTiler:
    def __init__(
        self,
        mappings: dict[Mapping, np.ndarray],
        config: FlowlineHatcherConfig,
        num_tiles: tuple[int, int],
    ):
        self.mappings = mappings
        self.config = config
        self.num_tiles = num_tiles

        self.tiles: list[list[dict[str, int | np.ndarray]]] = [
            [{} for _ in range(num_tiles[0])] for _ in range(num_tiles[1])
        ]

        self.row_size = int(self.mappings[Mapping.DISTANCE].shape[0] / self.num_tiles[1])
        self.col_size = int(self.mappings[Mapping.DISTANCE].shape[1] / self.num_tiles[0])

        for col in range(self.num_tiles[0]):
            for row in range(self.num_tiles[1]):
                self.tiles[row][col]["min_row"] = int(self.row_size * row)
                self.tiles[row][col]["min_col"] = int(self.col_size * col)
                self.tiles[row][col]["max_row"] = int(self.row_size * (row + 1))
                self.tiles[row][col]["max_col"] = int(self.col_size * (col + 1))

    def hatch(self) -> list[LineString]:
        for col in range(self.num_tiles[0]):
            for row in range(self.num_tiles[1]):
                logger.debug(f"processing tile {col} {row}")

                t = self.tiles[row][col]

                # extract first_stage_points from the point_raster of already processed neighbouring tiles
                initial_seed_points = []
                if self.config.COLLISION_APPROXIMATE:
                    if col > 0:
                        point_raster_left = self.tiles[row][col - 1]["hatcher"].point_raster
                        for y in point_raster_left[:, -2].nonzero()[0]:
                            initial_seed_points.append((0, y))

                    if row > 0:
                        point_raster_top = self.tiles[row - 1][col]["hatcher"].point_raster
                        for x in point_raster_top[-2, :].nonzero()[0]:
                            initial_seed_points.append((x, 0))
                else:
                    raise NotImplementedError()

                tile_mappings = [
                    self.mappings[Mapping.DISTANCE],
                    self.mappings[Mapping.ANGLE],
                    self.mappings[Mapping.MAX_LENGTH],
                    self.mappings[Mapping.FLAT],
                ]
                tile_mappings = [
                    mapping[t["min_row"] : t["max_row"], t["min_col"] : t["max_col"]] for mapping in tile_mappings
                ]

                hatcher = FlowlineHatcher(
                    [self.col_size, self.row_size],
                    tile_mappings,
                    self.config,
                    initial_seed_points=initial_seed_points,
                )

                linestrings = hatcher.hatch()
                linestrings = [
                    shapely.affinity.translate(ls, xoff=t["min_col"], yoff=t["min_row"]) for ls in linestrings
                ]

                self.tiles[row][col]["linestrings"] = linestrings
                self.tiles[row][col]["hatcher"] = hatcher

        linestrings = []
        for col in range(self.num_tiles[0]):
            for row in range(self.num_tiles[1]):
                linestrings += self.tiles[row][col]["linestrings"]

        return linestrings


class FlowlinesRust:
    def __init__(
        self,
        mappings: dict[Mapping, np.ndarray],
        config: FlowlineHatcherConfig,
        polygon: Polygon,
    ):
        self.mappings = mappings
        self.config = config
        self.polygon = polygon

    def hatch(self) -> list[LineString]:
        all_linestrings = []

        min_col, min_row, max_col, max_row = [int(e) for e in shapely.bounds(self.polygon).tolist()]
        max_col += 1
        max_row += 1

        tile_mappings = [
            self.mappings[Mapping.DISTANCE],
            self.mappings[Mapping.ANGLE],
            self.mappings[Mapping.MAX_LENGTH],
            self.mappings[Mapping.FLAT],
        ]
        tile_mappings = [mapping[min_row:max_row, min_col:max_col] for mapping in tile_mappings]

        dimensions = (max_col - min_col, max_row - min_row)

        rust_config = flowlines_py.FlowlinesConfig()
        rust_config = _py_config_to_rust_config(self.config, rust_config)

        rust_lines: list[list[tuple[float, float]]] = flowlines_py.hatch(dimensions, rust_config, *tile_mappings)
        linestrings = [shapely.affinity.translate(LineString(l), xoff=min_col, yoff=min_row) for l in rust_lines]

        linestrings_cropped = []
        for ls in linestrings:
            cropped = shapely.intersection(ls, self.polygon)

            if type(cropped) is Point:
                pass
            elif type(cropped) is MultiLineString:
                g = geometrytools.unpack_multilinestring(cropped)
                linestrings_cropped += g
            else:
                linestrings_cropped.append(cropped)

        all_linestrings += list(itertools.filterfalse(shapely.is_empty, linestrings_cropped))

        return all_linestrings


class FlowlineHatcher:
    """
    A Python implementation of the paper "Creating Evenly-Spaced Streamlines
    of Arbitrary Density" by Bruno Jobard and Wilfrid Lefer.

    """

    def __init__(
        self,
        dimensions: list[int] | tuple[int, int],
        mappings: list[np.ndarray],
        config: FlowlineHatcherConfig,
        initial_seed_points: list[tuple[float, float]] = [],
        tile_name: str = "",
    ):
        self.dimensions = dimensions
        self.config = config

        self.scale_x = mappings[0].shape[1] / dimensions[0]
        self.scale_y = mappings[0].shape[0] / dimensions[1]

        self.distance = mappings[0]
        self.angles = (mappings[1].astype(float) / 255.0) * math.tau - math.pi
        self.max_length = mappings[2]
        self.flat = mappings[3]

        self.initial_seed_points = initial_seed_points
        self.tile_name = tile_name

        if self.config.COLLISION_APPROXIMATE:
            self.MAPPING_FACTOR_COLLISION = int(math.ceil(1 / self.config.LINE_DISTANCE[0]))
            self.point_raster = np.zeros(
                [
                    self.dimensions[1] * self.MAPPING_FACTOR_COLLISION,
                    self.dimensions[0] * self.MAPPING_FACTOR_COLLISION,
                ],
                dtype=bool,
            )
        else:
            self.point_bins = []
            self.bin_size = self.config.LINE_DISTANCE[1]
            self.num_bins_x = int(self.dimensions[0] // self.bin_size + 1)
            self.num_bins_y = int(self.dimensions[1] // self.bin_size + 1)

            for x in range(self.num_bins_x):
                self.point_bins.append([np.empty([0, 2], dtype=float)] * self.num_bins_y)

    def _map_line_distance(self, x: float, y: float) -> float:
        return float(
            self.config.LINE_DISTANCE[0]
            + self.distance[int(y * self.scale_y), int(x * self.scale_x)]
            / 255
            * (self.config.LINE_DISTANCE[1] - self.config.LINE_DISTANCE[0])
        )

    def _map_angle(self, x: float, y: float) -> float:
        return float(self.angles[int(y * self.scale_y), int(x * self.scale_x)])

    def _map_line_max_length(self, x: float, y: float) -> float:
        return float(
            self.config.LINE_MAX_LENGTH[0]
            + self.max_length[int(y * self.scale_y), int(x * self.scale_x)]
            / 255
            * (self.config.LINE_MAX_LENGTH[1] - self.config.LINE_MAX_LENGTH[0])
        )

    def _map_flat(self, x: float, y: float) -> bool:
        return self.flat[int(y * self.scale_y), int(x * self.scale_x)] > 0

    def _collision_approximate(self, x: float, y: float, factor: float) -> bool:
        if x >= self.dimensions[0]:
            return True

        if y >= self.dimensions[1]:
            return True

        min_d = int(self._map_line_distance(x, y) * factor * self.MAPPING_FACTOR_COLLISION)

        rm_x = int(x * self.MAPPING_FACTOR_COLLISION)
        rm_y = int(y * self.MAPPING_FACTOR_COLLISION)

        return np.any(
            self.point_raster[
                max(rm_y - min_d, 0) : rm_y + min_d + 1,
                max(rm_x - min_d, 0) : rm_x + min_d + 1,
            ]
        )

    def _collision_precise(self, x: float, y: float, factor: float) -> bool:
        min_d = self._map_line_distance(x, y) * factor
        bin_pos = [int(x / self.bin_size), int(y / self.bin_size)]

        bins = [
            [bin_pos[0], bin_pos[1] - 1],
            [bin_pos[0] - 1, bin_pos[1]],
            [bin_pos[0], bin_pos[1]],
            [bin_pos[0] + 1, bin_pos[1]],
            [bin_pos[0], bin_pos[1] + 1],
        ]

        for ix, iy in bins:
            if ix < 0 or ix >= self.num_bins_x:
                continue

            if iy < 0 or iy >= self.num_bins_y:
                continue

            arr = self.point_bins[ix][iy]

            if arr.shape[0] == 0:
                continue

            distance = np.linalg.norm(arr - np.array([x, y]), axis=1)
            if np.any(distance < min_d):
                return True

        return False

    def _collision(self, x: float, y: float, factor: float = 1.0) -> bool:
        if self.config.COLLISION_APPROXIMATE:
            return self._collision_approximate(x, y, factor)
        else:
            return self._collision_precise(x, y, factor)

    def _next_point(self, x1: float, y1: float, forwards: bool) -> tuple[float, float] | None:
        a1 = self._map_angle(x1, y1)

        if self._map_flat(x1, y1):
            return None

        dir = 1
        if not forwards:
            dir = -1

        x2 = x1 + self.config.LINE_STEP_DISTANCE * math.cos(a1) * dir
        y2 = y1 + self.config.LINE_STEP_DISTANCE * math.sin(a1) * dir

        if x2 < 0 or x2 >= self.dimensions[0] or y2 < 0 or y2 >= self.dimensions[1]:
            return None

        if self._collision(x2, y2, factor=self.config.LINE_DISTANCE_END_FACTOR):
            return None

        if self.config.MAX_ANGLE_DISCONTINUITY > 0:
            a2 = self._map_angle(x2, y2)

            if abs(a2 - a1) > self.config.MAX_ANGLE_DISCONTINUITY:
                # print("MAX_ANGLE_DISCONTINUITY")
                return None

        return (x2, y2)

    def _seed_points(self, line_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        num_seedpoints = 1
        seed_points = []

        if len(line_points) > self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS:
            num_seedpoints = (len(line_points) - 1) // self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS

        for i in range(num_seedpoints):
            x1, y1 = line_points[i * self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS]
            x2, y2 = line_points[i * self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS + 1]

            # midpoint
            x3 = x1 + (x2 - x1) / 2.0
            y3 = y1 + (y2 - y1) / 2.0

            a1 = math.atan2(y1 - y3, x1 - x3)

            a2 = a1
            if i % 2 == 0:
                a2 += math.radians(90)
            else:
                a2 -= math.radians(90)

            x4 = self._map_line_distance(x3, y3)
            y4 = 0

            x5 = x4 * math.cos(a2) - y4 * math.sin(a2) + x3
            y5 = x4 * math.sin(a2) + y4 * math.cos(a2) + y3

            if x5 < 0 or x5 >= self.dimensions[0] or y5 < 0 or y5 >= self.dimensions[1]:
                continue

            seed_points.append((x5, y5))

        return seed_points

    def hatch(self) -> list[LineString]:
        linestrings = []
        starting_points = deque()
        starting_points_priority = deque(self.initial_seed_points)

        # point grid for starting points, grid distance is mean line distance
        num_gridpoints_x = int(self.dimensions[0] / (self.config.LINE_DISTANCE[0] * 1.5))
        num_gridpoints_y = int(self.dimensions[1] / (self.config.LINE_DISTANCE[0] * 1.5))

        for i in np.linspace(1, self.dimensions[0] - 1, endpoint=False, num=num_gridpoints_x):
            for j in np.linspace(1, self.dimensions[1] - 1, endpoint=False, num=num_gridpoints_y):
                starting_points.append([i, j])

        for i in range(MAX_ITERATIONS):
            if i >= MAX_ITERATIONS - 1:
                if len(self.tile_name) > 0:
                    logger.warning(f"{self.tile_name}: max iterations exceeded")
                else:
                    logger.warning("max iterations exceeded")

            if len(starting_points) == 0:
                break

            seed = None
            if len(starting_points_priority) > 0:
                seed = starting_points_priority.popleft()
            else:
                seed = starting_points.popleft()

            if self._collision(*seed):
                continue

            line_points = deque([seed])

            # follow gradient upwards
            for _ in range(int(self.config.LINE_MAX_LENGTH[1] / self.config.LINE_STEP_DISTANCE)):
                p = self._next_point(*line_points[-1], True)

                if p is None:
                    break

                if len(line_points) * self.config.LINE_STEP_DISTANCE > self._map_line_max_length(int(p[0]), int(p[1])):
                    break

                line_points.append(p)

            # follow gradient downwards
            for _ in range(int(self.config.LINE_MAX_LENGTH[1] / self.config.LINE_STEP_DISTANCE)):
                p = self._next_point(*line_points[0], False)

                if p is None:
                    break

                if len(line_points) * self.config.LINE_STEP_DISTANCE > self._map_line_max_length(int(p[0]), int(p[1])):
                    break

                line_points.appendleft(p)

            if len(line_points) < 2:
                continue

            linestrings.append(LineString(line_points))

            # seed points
            starting_points.extendleft(self._seed_points(line_points))

            # collision checks
            for lp in line_points:
                x = int(lp[0])
                y = int(lp[1])
                if self.config.COLLISION_APPROXIMATE:
                    self.point_raster[
                        int(y * self.MAPPING_FACTOR_COLLISION), int(x * self.MAPPING_FACTOR_COLLISION)
                    ] = True
                else:
                    # self.point_map[f"{x},{y}"].append(lp)
                    # self.point_bins[int(x/self.bin_size)][int(y/self.bin_size)].append(lp)
                    self.point_bins[int(x / self.bin_size)][int(y / self.bin_size)] = np.append(
                        self.point_bins[int(x / self.bin_size)][int(y / self.bin_size)], [lp], axis=0
                    )

        return linestrings


def _prepare_mappings(OUTPUT_PATH, RESIZE_SIZE) -> dict[Mapping, np.ndarray]:
    ELEVATION_FILE = Path("experiments/hatching/data/gebco_crop.tif")

    config = FlowlineHatcherConfig()

    elevation = None
    with rasterio.open(str(ELEVATION_FILE)) as dataset:
        elevation = dataset.read()[0]

    elevation = cv2.resize(elevation, RESIZE_SIZE)

    elevation[elevation > 0] = 0  # bathymetry data only

    _, _, _, _, angles, inclination = get_slope(elevation, 1)

    WINDOW_SIZE = 25
    MAX_WIN_VAR = 40000
    win_mean = ndimage.uniform_filter(elevation.astype(float), (WINDOW_SIZE, WINDOW_SIZE))
    win_sqr_mean = ndimage.uniform_filter(elevation.astype(float) ** 2, (WINDOW_SIZE, WINDOW_SIZE))
    win_var = win_sqr_mean - win_mean**2
    win_var = np.clip(win_var, 0, MAX_WIN_VAR)
    win_var = win_var * -1 + MAX_WIN_VAR
    win_var = normalize_to_uint8(win_var)

    # uint8 image must be centered around 128 to deal with negative values
    mapping_angle = ((angles + math.pi) / math.tau * 255.0).astype(np.uint8)
    mapping_flat = np.zeros_like(inclination, dtype=np.uint8)
    mapping_flat[inclination < config.MIN_INCLINATION] = 255  # uint8
    mapping_distance = normalize_to_uint8(elevation)  # uint8
    mapping_max_length = win_var  # uint8

    mapping_angle = cv2.blur(mapping_angle, (40, 40))
    mapping_distance = cv2.blur(mapping_distance, (10, 10))
    mapping_max_length = cv2.blur(mapping_max_length, (10, 10))

    cv2.imwrite(str(Path(OUTPUT_PATH, "mapping_angle.png")), normalize_to_uint8(mapping_angle / math.tau))
    cv2.imwrite(str(Path(OUTPUT_PATH, "mapping_flat.png")), mapping_flat.astype(np.uint8) * 255)
    cv2.imwrite(str(Path(OUTPUT_PATH, "mapping_distance.png")), mapping_distance)
    cv2.imwrite(str(Path(OUTPUT_PATH, "mapping_max_segments.png")), mapping_max_length)

    mappings = {
        Mapping.DISTANCE: mapping_distance,
        Mapping.ANGLE: mapping_angle,
        Mapping.MAX_LENGTH: mapping_max_length,
        Mapping.FLAT: mapping_flat,
    }

    return mappings


if __name__ == "__main__":
    OUTPUT_PATH = Path(".")
    RESIZE_SIZE = (500, 500)

    import cProfile

    timer_total_runtime = datetime.datetime.now()

    pr = cProfile.Profile()
    pr.enable()

    mappings = _prepare_mappings(OUTPUT_PATH, RESIZE_SIZE)
    config = FlowlineHatcherConfig()
    config.COLLISION_APPROXIMATE = True
    config.LINE_DISTANCE = [0.3, 10.0]
    config.LINE_MAX_LENGTH = (50, 200)

    # tiler = FlowlineTiler(mappings, config, (2, 2))
    tiler = FlowlineTilerPoly(
        mappings,
        config,
        [Point([RESIZE_SIZE[0] // 2, RESIZE_SIZE[0] // 2]).buffer(min(RESIZE_SIZE) * 0.49)],
    )
    linestrings = tiler.hatch()

    pr.disable()
    pr.dump_stats("profile.pstat")
    logger.info(f"total time: {(datetime.datetime.now() - timer_total_runtime).total_seconds():5.2f}s")

    svg_path = Path(OUTPUT_PATH, "flowlines.svg")
    svg = SvgWriter(svg_path, RESIZE_SIZE)
    options = {"fill": "none", "stroke": "black", "stroke-width": "1"}

    svg.add("flowlines", linestrings, options=options)
    svg.write()

    try:
        convert_svg_to_png(svg, svg.dimensions[0] * 10)
    except Exception as e:
        logger.warning(f"SVG to PNG conversion failed: {e}")
