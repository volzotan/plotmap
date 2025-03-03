import argparse
import datetime
import itertools
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import rasterio
import shapely
from dask.distributed import LocalCluster
from loguru import logger
from matplotlib import pyplot as plt
from scipy import ndimage
from shapely import LineString, Polygon, Point, MultiLineString

from experiments.hatching import scales
from experiments.hatching.slope import get_slope
from lineworld.core.svgwriter import SvgWriter
from lineworld.util import geometrytools
from lineworld.util.export import convert_svg_to_png

MAPPING_DISTANCE = 0
MAPPING_MAX_SEGMENTS = 1


@dataclass
class FlowlineHatcherConfig:
    # distance between lines in mm
    LINE_DISTANCE: tuple[float, float] = (0.3, 5.0)
    LINE_DISTANCE_END_FACTOR = 0.5

    # distance between points constituting a line in mm
    LINE_STEP_DISTANCE: float = 0.2

    MM_TO_PX_CONVERSION_FACTOR: int = 10

    # max difference (in radians) in slope between line points
    MAX_ANGLE_DISCONTINUITY: float = math.pi / 2
    MIN_INCLINATION: float = 0.001  # 50.0

    # How many line segments should be skipped before the next seedpoint is extracted
    SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: int = 5

    LINE_MAX_SEGMENTS: tuple[int, int] = (10, 50)

    BLUR_ANGLES: bool = True
    BLUR_ANGLES_KERNEL_SIZE: int = 40

    BLUR_INCLINATION: bool = True
    BLUR_INCLINATION_KERNEL_SIZE: int = 10

    BLUR_MAPPING_DISTANCE: bool = True
    BLUR_MAPPING_DISTANCE_KERNEL_SIZE: int = 10

    SCALE_ADJUSTMENT_VALUE: float = 0.3

    COLLISION_APPROXIMATE: bool = True
    VIZ_LINE_THICKNESS: int = 5


class FlowlineTilerPoly:
    def __init__(
        self,
        elevation: np.ndarray,
        mappings: np.ndarray,
        config: FlowlineHatcherConfig,
        polygons: list[Polygon],
    ):
        self.elevation = elevation
        self.mappings = mappings
        self.config = config
        self.polygons = polygons

        # sanity checks
        if self.config.COLLISION_APPROXIMATE:
            if self.config.MM_TO_PX_CONVERSION_FACTOR * self.config.LINE_DISTANCE[0] < 1:
                raise Exception("elevation raster data too coarse for LINE_DISTANCE config settings")

            if self.config.MM_TO_PX_CONVERSION_FACTOR * self.config.LINE_STEP_DISTANCE < 1:
                raise Exception("elevation raster data too coarse for LINE_DISTANCE config settings")

        # Prepare a non-linear scale for the density calculations
        scale_adjustment_value = self.config.SCALE_ADJUSTMENT_VALUE
        scale = scales.Scale(
            scales.quadratic_bezier,
            {
                "p1": [0.0 + scale_adjustment_value, 0.0],
                "p2": [1.0 - scale_adjustment_value, 1.0],
            },
        )

        self.tiles = [{} for _ in polygons]

    def hatch(self) -> list[LineString]:
        cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit="6GB")
        client = cluster.get_client()

        # import coiled
        # cluster = coiled.Cluster(n_workers=3)
        # client = cluster.get_client()

        futures = []

        # no dask
        # 2025-01-16 19:46:17.344 | DEBUG    | __main__:run:86 - draw in 867.52s

        # dask gets a complete, preinitialized hatcher and executes hatch()
        # 2025-01-16 18:36:39.800 | DEBUG    | __main__:run:86 - draw in 419.25s

        # dask gets hatcher.hatch() containing only a view on elevation and executes hatch()
        # 2025-01-16 18:49:40.896 | DEBUG    | __main__:run:86 - draw in 328.34s

        # dask gets a hatcher containing only a view on elevation and executes hatch() and does the intersection cropping
        # 2025-01-16 19:08:09.530 | DEBUG    | __main__:run:86 - draw in 356.36s

        # dask gets a hatch func containing only a view on elevation and executes hatch() and does the intersection cropping
        # 2025-01-16 19:26:34.196 | DEBUG    | __main__:run:86 - draw in 383.84s

        # dask gets a hatch func containing only a view on elevation and executes hatch() and does the intersection cropping
        # + reduced file size by switching from float64 to uint8
        # 2025-01-16 21:44:00.881 | DEBUG    | __main__:run:86 - draw in 324.53s

        # coiled, 3 workers
        # 2025-01-16 22:49:54.305 | DEBUG    | __main__:run:86 - draw in 608.36s

        for i, p in enumerate(self.polygons):
            logger.debug(f"processing tile {i:03}/{len(self.polygons):03} : {i/len(self.polygons)*100.0:5.2f}%")

            min_col, min_row, max_col, max_row = [int(e) for e in shapely.bounds(p).tolist()]
            max_col += 1
            max_row += 1

            if max_row - min_row < 10 or max_col - min_col < 10:
                logger.warning(f"empty tile {p}")
                self.tiles[i]["linestrings"] = []
                continue

            elevation_tile = self.elevation[min_row:max_row, min_col:max_col]
            mappings_tile = self.mappings[min_row:max_row, min_col:max_col, :]

            def compute(
                p: Polygon,
                crop: np.ndarray,
                mappings: np.ndarray,
                config: FlowlineHatcherConfig,
                xoff: float,
                yoff: float,
            ) -> list[LineString]:
                hatcher = FlowlineHatcher(p, crop, mappings, config)
                linestrings = hatcher.hatch()
                del hatcher
                linestrings = [shapely.affinity.translate(ls, xoff=xoff, yoff=yoff) for ls in linestrings]

                linestrings_cropped = []
                for ls in linestrings:
                    cropped = shapely.intersection(ls, p)

                    if type(cropped) is Point:
                        pass
                    elif type(cropped) is MultiLineString:
                        g = geometrytools.unpack_multilinestring(cropped)
                        linestrings_cropped += g
                    else:
                        linestrings_cropped.append(cropped)

                linestrings = list(itertools.filterfalse(shapely.is_empty, linestrings_cropped))

                return linestrings

            futures.append(
                client.submit(
                    compute,
                    p,
                    np.copy(elevation_tile),
                    np.copy(mappings_tile),
                    self.config,
                    min_col,
                    min_row,
                )
            )

        for i, f in enumerate(futures):
            self.tiles[i]["linestrings"] = f.result()

        linestrings = []
        for tile in self.tiles:
            linestrings += tile["linestrings"]

        return linestrings

    def _debug_viz(self, linestrings: list[LineString]) -> None:
        output = np.full([self.elevation.shape[0], self.elevation.shape[1], 3], 255, dtype=np.uint8)
        for ls in linestrings:
            line_points = ls.coords
            for i in range(len(line_points) - 1):
                start = (int(line_points[i][0]), int(line_points[i][1]))
                end = (int(line_points[i + 1][0]), int(line_points[i + 1][1]))
                cv2.line(output, start, end, (0, 0, 0), self.config.VIZ_LINE_THICKNESS)

        output2 = np.full(
            [self.elevation.shape[0], self.elevation.shape[1] * 2, 3],
            255,
            dtype=np.uint8,
        )

        output2[:, 0 : self.elevation.shape[1], :] = output

        scale, norm_min, norm_max = self.density_func
        density_normalized = (self.elevation - norm_min) / (norm_max - norm_min)
        density_normalized = scale(density_normalized)
        density_normalized *= 255

        output2[:, self.elevation.shape[1] :, :] = np.dstack(
            [density_normalized, density_normalized, density_normalized]
        )

        # divider line
        output2[:, self.elevation.shape[1] : self.elevation.shape[1] + 2, :] = [0, 0, 0]

        cv2.imwrite(str(Path(OUTPUT_PATH, "flowlines.png")), output2)


class FlowlineTiler:
    def __init__(
        self,
        elevation: np.ndarray,
        mappings: np.ndarray,
        config: FlowlineHatcherConfig,
        num_tiles: tuple[int, int],
    ):
        self.elevation = elevation
        self.mappings = mappings
        self.config = config
        self.num_tiles = num_tiles

        # sanity checks
        if self.config.COLLISION_APPROXIMATE:
            if self.config.MM_TO_PX_CONVERSION_FACTOR * self.config.LINE_DISTANCE[0] < 1:
                raise Exception("elevation raster data too coarse for LINE_DISTANCE config settings")

            if self.config.MM_TO_PX_CONVERSION_FACTOR * self.config.LINE_STEP_DISTANCE < 1:
                raise Exception("elevation raster data too coarse for LINE_DISTANCE config settings")

        # Prepare a non-linear scale for the density calculations
        scale_adjustment_value = self.config.SCALE_ADJUSTMENT_VALUE
        scale = scales.Scale(
            scales.quadratic_bezier,
            {
                "p1": [0.0 + scale_adjustment_value, 0.0],
                "p2": [1.0 - scale_adjustment_value, 1.0],
            },
        )

        self.tiles: list[list[dict[str, int | np.ndarray]]] = [
            [{} for _ in range(num_tiles[0])] for _ in range(num_tiles[1])
        ]

        self.row_size = int(self.elevation.shape[0] / self.num_tiles[1])
        self.col_size = int(self.elevation.shape[1] / self.num_tiles[0])

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
                            initial_seed_points.append([0, y])

                    if row > 0:
                        point_raster_top = self.tiles[row - 1][col]["hatcher"].point_raster
                        for x in point_raster_top[-2, :].nonzero()[0]:
                            initial_seed_points.append([x, 0])
                else:
                    pass  # TODO

                elevation_tile = self.elevation[t["min_row"] : t["max_row"], t["min_col"] : t["max_col"]]
                mappings_tile = self.mappings[t["min_row"] : t["max_row"], t["min_col"] : t["max_col"], :]

                hatcher = FlowlineHatcher(
                    shapely.box(0, 0, self.col_size, self.row_size),
                    elevation_tile,
                    mappings_tile,
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

    def _debug_viz(self, linestrings: list[LineString]) -> None:
        output = np.full([self.elevation.shape[0], self.elevation.shape[1], 3], 255, dtype=np.uint8)
        for ls in linestrings:
            line_points = ls.coords
            for i in range(len(line_points) - 1):
                start = (int(line_points[i][0]), int(line_points[i][1]))
                end = (int(line_points[i + 1][0]), int(line_points[i + 1][1]))
                cv2.line(output, start, end, (0, 0, 0), self.config.VIZ_LINE_THICKNESS)

        # output[point_raster, :] = [0, 0, 255]

        angles = np.zeros(self.elevation.shape, dtype=np.float64)
        inclination = np.zeros(self.elevation.shape, dtype=np.uint8)
        point_raster = np.zeros(self.elevation.shape, dtype=np.uint8)

        for col in range(self.num_tiles[0]):
            for row in range(self.num_tiles[1]):
                t = self.tiles[row][col]
                angles[t["min_row"] : t["max_row"], t["min_col"] : t["max_col"]] = t["hatcher"].angles
                inclination[t["min_row"] : t["max_row"], t["min_col"] : t["max_col"]] = t["hatcher"].inclination
                if config.COLLISION_APPROXIMATE:
                    point_raster[t["min_row"] : t["max_row"], t["min_col"] : t["max_col"]] = t["hatcher"].point_raster

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

        ax1.imshow(self.elevation, cmap="binary")
        ax1.set_title("elevation")

        ax2.imshow(angles, cmap="hsv")
        ax2.set_title("angles")

        ax3.imshow(inclination)
        ax3.set_title("inclination")

        if self.density is not None:
            ax4.imshow(self.density, cmap="gray")
        else:
            scale, norm_min, norm_max = self.density_func
            density_normalized = (self.elevation - norm_min) / (norm_max - norm_min)
            density_normalized = scale(density_normalized)
            ax4.imshow(density_normalized, cmap="gray")
        ax4.set_title("density")

        ax5.imshow(output)
        ax5.set_title("output")

        ax6.imshow(point_raster)
        ax6.set_title("collision map")

        fig.set_figheight(12)
        fig.set_figwidth(20)
        ax2.get_yaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)
        ax6.get_yaxis().set_visible(False)

        plt.tight_layout = True

        plt.savefig(Path(OUTPUT_PATH, "flowlines_overview.png"))

        # cv2.imwrite(Path(OUTPUT_PATH, "flowlines.png"), output)

        # output_filename = f"flowlines_BLURANGLES-{BLUR_ANGLES}_BLURDENSITY-{BLUR_DENSITY_MAP}.png"
        # cv2.imwrite(Path(OUTPUT_PATH, output_filename), output)

        # cv2.circle(output, (round(seed[0]), round(seed[1])), 2, (255, 0, 0), -1)
        # for i in range(len(line_points) - 1):
        #     start = (round(line_points[i][0]), round(line_points[i][1]))
        #     end = (round(line_points[i + 1][0]), round(line_points[i + 1][1]))
        #     cv2.line(output, start, end, (255, 255, 255), 2)

        output2 = np.full(
            [self.elevation.shape[0], self.elevation.shape[1] * 2, 3],
            255,
            dtype=np.uint8,
        )

        output2[:, 0 : self.elevation.shape[1], :] = output

        scale, norm_min, norm_max = self.density_func
        density_normalized = (self.elevation - norm_min) / (norm_max - norm_min)
        density_normalized = scale(density_normalized)
        density_normalized *= 255

        output2[:, self.elevation.shape[1] :, :] = np.dstack(
            [density_normalized, density_normalized, density_normalized]
        )

        # divider line
        output2[:, self.elevation.shape[1] : self.elevation.shape[1] + 2, :] = [0, 0, 0]

        cv2.imwrite(str(Path(OUTPUT_PATH, "flowlines.png")), output2)


class FlowlineHatcher:
    MAX_ITERATIONS = 20_000_000

    def __init__(
        self,
        polygon: Polygon,
        elevation: np.ndarray,
        mappings: np.ndarray,
        config: FlowlineHatcherConfig,
        initial_seed_points: list[tuple[float, float]] = [],
        tile_name: str = "",
    ):
        self.polygon = polygon
        self.elevation = elevation
        self.mappings = mappings
        self.config = config
        self.initial_seed_points = initial_seed_points
        self.tile_name = tile_name

        self.bbox = self.polygon.bounds
        self.bbox = [
            0,
            0,
            math.ceil(self.bbox[2] - self.bbox[0]),
            math.ceil(self.bbox[3] - self.bbox[1]),
        ]  # minx, miny, maxx, maxy

        if self.config.COLLISION_APPROXIMATE:
            self.point_raster = np.zeros([self.bbox[3], self.bbox[2]], dtype=bool)
        else:
            self.point_map = {}
            for x in range(self.bbox[2] + 1):
                for y in range(self.bbox[3] + 1):
                    self.point_map[f"{x},{y}"] = []

    def _lazy_init(self):
        _, _, _, _, angles, inclination = get_slope(self.elevation, 1)

        self.angles = angles
        self.inclination = inclination

        # if the mappings are not a normalized float array (for example to save
        # memory when using dask), we need to normalize it now
        if self.mappings.dtype not in [
            float,
            np.float16,
            np.float32,
            np.float64,
        ]:
            self.mappings = self.mappings / np.iinfo(self.mappings.dtype).max

        if self.config.BLUR_ANGLES:
            self.angles = cv2.blur(
                self.angles,
                (
                    self.config.BLUR_ANGLES_KERNEL_SIZE,
                    self.config.BLUR_ANGLES_KERNEL_SIZE,
                ),
            )

        if self.config.BLUR_INCLINATION:
            self.inclination = cv2.blur(
                self.inclination,
                (
                    self.config.BLUR_INCLINATION_KERNEL_SIZE,
                    self.config.BLUR_INCLINATION_KERNEL_SIZE,
                ),
            )

        if self.config.BLUR_MAPPING_DISTANCE:
            self.mappings[:, :, MAPPING_DISTANCE] = cv2.blur(
                self.mappings[:, :, MAPPING_DISTANCE],
                (
                    self.config.BLUR_MAPPING_DISTANCE_KERNEL_SIZE,
                    self.config.BLUR_MAPPING_DISTANCE_KERNEL_SIZE,
                ),
            )

    def _collision_approximate(self, x: float, y: float, factor: float) -> bool:
        x = int(x)
        y = int(y)

        if y >= self.point_raster.shape[0]:
            return True
        if x >= self.point_raster.shape[1]:
            return True

        half_d = int((self._map_line_distance(x, y) / 2) * factor)

        return np.any(
            self.point_raster[
                max(y - half_d, 0) : min(y + half_d + 1, self.point_raster.shape[0]),
                max(x - half_d, 0) : min(x + half_d + 1, self.point_raster.shape[1]),
            ]
        )

    def _map_line_distance(self, x: int, y: int) -> float:
        diff = (self.config.LINE_DISTANCE[1] - self.config.LINE_DISTANCE[0]) * self.config.MM_TO_PX_CONVERSION_FACTOR
        return float(
            self.config.LINE_DISTANCE[0] * self.config.MM_TO_PX_CONVERSION_FACTOR
            + self.mappings[y, x, MAPPING_DISTANCE] * diff
        )

    def _map_line_segments(self, x: int, y: int) -> float:
        diff = self.config.LINE_MAX_SEGMENTS[1] - self.config.LINE_MAX_SEGMENTS[0]
        return float(self.config.LINE_MAX_SEGMENTS[0] + self.mappings[y, x, MAPPING_MAX_SEGMENTS] * diff)

    def _collision_precise(self, x: float, y: float, factor: float) -> bool:
        # rx = round(x)
        # ry = round(y)
        # # d = self._map_line_distance[ry, rx]
        # d = self._map_line_distance(rx, ry)

        rx = int(x)
        ry = int(y)
        d = self._map_line_distance(rx, ry) * factor
        half_d = math.ceil(d / 2)

        x_minmax = [max(rx - half_d, 0), min(rx + half_d, self.elevation.shape[1])]
        y_minmax = [max(ry - half_d, 0), min(ry + half_d, self.elevation.shape[0])]

        for ix in range(*x_minmax):
            for iy in range(*y_minmax):
                for p in self.point_map[f"{ix},{iy}"]:
                    if math.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2) < d:
                        return True

        return False

    def _collision(self, x: float, y: float, factor: float = 1.0) -> bool:
        if self.config.COLLISION_APPROXIMATE:
            return self._collision_approximate(x, y, factor)
        else:
            return self._collision_precise(x, y, factor)

    def _next_point(self, x1: float, y1: float, forwards: bool) -> tuple[float, float] | None:
        rx1 = int(x1)
        ry1 = int(y1)

        a1 = self.angles[ry1, rx1]
        inc = self.inclination[ry1, rx1]

        if abs(inc) < self.config.MIN_INCLINATION:
            return None

        dir = 1
        if not forwards:
            dir = -1

        x2 = x1 + self.config.LINE_STEP_DISTANCE * self.config.MM_TO_PX_CONVERSION_FACTOR * math.cos(a1) * dir
        y2 = y1 + self.config.LINE_STEP_DISTANCE * self.config.MM_TO_PX_CONVERSION_FACTOR * math.sin(a1) * dir

        # if not self.polygon.contains(Point(x2, y2)):
        #     return None

        if x2 < 0 or x2 > self.bbox[2] or y2 < 0 or y2 > self.bbox[3]:  # TODO
            return None

        if self._collision(x2, y2, factor=self.config.LINE_DISTANCE_END_FACTOR):
            return None

        if self.config.MAX_ANGLE_DISCONTINUITY > 0:
            a2 = self.angles[int(y2), int(x2)]

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

            # x4 = self.density[int(y3), int(x3)]
            x4 = self._map_line_distance(int(x3), int(y3))
            y4 = 0

            x5 = x4 * math.cos(a2) - y4 * math.sin(a2) + x3
            y5 = x4 * math.sin(a2) + y4 * math.cos(a2) + y3

            # if not self.polygon.contains(Point([x5, y5])):
            #     continue

            if x5 < 0 or x5 > self.bbox[2] or y5 < 0 or y5 > self.bbox[3]:  # TODO
                continue

            seed_points.append((x5, y5))

        return seed_points

    def hatch(self) -> list[LineString]:
        self._lazy_init()

        linestrings = []
        starting_points = deque()
        starting_points_priority = deque(self.initial_seed_points)

        # point grid for starting points, grid distance is mean line distance
        num_gridpoints = int(
            (self.elevation.shape[0] / self.config.MM_TO_PX_CONVERSION_FACTOR) / self.config.LINE_DISTANCE[0] * 2
        )  # np.mean(self.config.LINE_DISTANCE))
        for i in np.linspace(self.bbox[0] + 1, self.bbox[2] - 1, num=num_gridpoints):
            for j in np.linspace(self.bbox[1] + 1, self.bbox[3] - 1, num=num_gridpoints):
                starting_points.append([i, j])

        for i in range(self.MAX_ITERATIONS):

            if i >= self.MAX_ITERATIONS - 1:
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
            for _ in range(self.config.LINE_MAX_SEGMENTS[1]):
                p = self._next_point(*line_points[-1], True)

                if p is None:
                    break

                if len(line_points) > self._map_line_segments(int(p[0]), int(p[1])):
                    break

                line_points.append(p)

            # follow gradient downwards
            for _ in range(self.config.LINE_MAX_SEGMENTS[1]):
                p = self._next_point(*line_points[0], False)

                if p is None:
                    break

                if len(line_points) > self._map_line_segments(int(p[0]), int(p[1])):
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
                    self.point_raster[y, x] = True
                else:
                    self.point_map[f"{x},{y}"].append(lp)

        return linestrings


# ELEVATION_FILE = Path("experiments/hatching/data/GebcoToBlender/fullsize_reproject.tif")
# # ELEVATION_FILE = Path("experiments/hatching/data/gebco_crop.tif")
# # DENSITY_FILE = Path("shaded_relief4.png")
# DENSITY_FILE = ELEVATION_FILE
#
# TWO_TONE_FILE = Path("experiments/hatching/data/two_tone_blender.png")
# TWO_TONE_FILE = Path("experiments/hatching/data/two_tone_blender_blurred.png")
# TWO_TONE_FILE = Path("experiments/hatching/data/two_tone_blender_2.png")
#
# OUTPUT_PATH = Path("experiments/hatching/output")

if __name__ == "__main__":
    ELEVATION_FILE = Path("experiments/hatching/data/gebco_crop.tif")
    OUTPUT_PATH = Path("experiments/hatching/output")
    RESIZE_SIZE = (3000, 3000)

    timer_total_runtime = datetime.datetime.now()

    data = None
    with rasterio.open(str(ELEVATION_FILE)) as dataset:
        # data = dataset.read(
        #     out_shape=RESIZE_SIZE,
        #     resampling=rasterio.enums.Resampling.bilinear,
        # )[0]

        data = dataset.read()[0]

    data = cv2.resize(data, RESIZE_SIZE)

    # CROP_CENTER = [.5,.5] # [0.4, 0.4]
    # CROP_SIZE = [10000, 10000]
    # data = data[
    #        int(CROP_CENTER[1] * data.shape[1] - CROP_SIZE[1] // 2): int(
    #            CROP_CENTER[1] * data.shape[1] + CROP_SIZE[1] // 2),
    #        int(CROP_CENTER[0] * data.shape[0] - CROP_SIZE[0] // 2): int(
    #            CROP_CENTER[0] * data.shape[0] + CROP_SIZE[0] // 2),
    #        ]

    elevation_normalized = (np.iinfo(np.uint8).max * ((data - np.min(data)) / np.ptp(data))).astype(np.uint8)

    mappings = np.zeros([data.shape[0], data.shape[1], 2], dtype=np.uint8)
    mappings[:, :, MAPPING_DISTANCE] = elevation_normalized[:, :]

    WINDOW_SIZE = 25
    MAX_WIN_VAR = 40000
    win_mean = ndimage.uniform_filter(data.astype(float), (WINDOW_SIZE, WINDOW_SIZE))
    win_sqr_mean = ndimage.uniform_filter(data.astype(float) ** 2, (WINDOW_SIZE, WINDOW_SIZE))
    win_var = win_sqr_mean - win_mean**2
    win_var = np.clip(win_var, 0, MAX_WIN_VAR)
    win_var = win_var * -1 + MAX_WIN_VAR
    win_var = (np.iinfo(np.uint8).max * ((win_var - np.min(win_var)) / np.ptp(win_var))).astype(np.uint8)

    mappings[:, :, MAPPING_MAX_SEGMENTS] = win_var

    # mappings[:,:, MAPPING_MAX_SEGMENTS] = np.full_like(mappings[:,:, MAPPING_MAX_SEGMENTS], 255)

    cv2.imwrite(str(Path(OUTPUT_PATH, "MAPPING_DISTANCE.png")), mappings[:, :, MAPPING_DISTANCE])
    cv2.imwrite(str(Path(OUTPUT_PATH, "MAPPING_MAX_SEGMENTS.png")), mappings[:, :, MAPPING_MAX_SEGMENTS])

    config = FlowlineHatcherConfig()
    tiler = FlowlineTiler(data, mappings, config, (2, 2))
    linestrings = tiler.hatch()

    svg_path = Path(OUTPUT_PATH, "flowlines.svg")
    svg = SvgWriter(svg_path, data.shape)
    options = {"fill": "none", "stroke": "black", "stroke-width": "2"}

    svg.add("flowlines", linestrings, options=options)
    svg.write()

    try:
        convert_svg_to_png(svg_path, svg.dimensions[0] * 10)
    except Exception as e:
        logger.warning(f"SVG to PNG conversion failed: {e}")

    logger.info(f"total time: {(datetime.datetime.now() - timer_total_runtime).total_seconds():5.2f}s")

    exit()

    # --------

    # RESIZE_SIZE = [20_000, 20_000]
    CROP_SIZE = [10000, 10000]

    parser = argparse.ArgumentParser()
    parser.add_argument("--resize", type=int)
    parser.add_argument("--BLUR_ANGLES_KERNEL_SIZE", type=int)
    args = vars(parser.parse_args())

    config = FlowlineHatcherConfig()

    config.BLUR_ANGLES_KERNEL_SIZE = 50
    config.BLUR_DENSITY_KERNEL_SIZE = 50
    config.BLUR_INCLINATION_KERNEL_SIZE = 20

    config.LINE_MAX_SEGMENTS = 30  # 6
    config.LINE_DISTANCE = (0.3, 3.0)

    if args["BLUR_ANGLES_KERNEL_SIZE"] is not None:
        config.BLUR_ANGLES_KERNEL_SIZE = args["BLUR_ANGLES_KERNEL_SIZE"]

    # # since we are working in the test environment directly on the raster image coordinate space
    # # and not with map coordinates that will be exported in a SVG, we need to scale the millimeter-values to raster-space
    # config.PX_PER_MM = 5
    # config.LINE_DISTANCE = [e * config.PX_PER_MM for e in config.LINE_DISTANCE]
    # config.LINE_STEP_DISTANCE += config.PX_PER_MM

    data = None
    with rasterio.open(str(ELEVATION_FILE)) as dataset:
        if args["resize"] is not None:
            data = dataset.read(
                out_shape=(args["resize"], args["resize"]),
                resampling=rasterio.enums.Resampling.bilinear,
            )[0]
        else:
            data = dataset.read(1)

    # data = cv2.resize(data, RESIZE_SIZE)

    CROP_X, CROP_Y = [data.shape[0] // 2, data.shape[1] // 2]
    data = data[
        CROP_Y - CROP_SIZE[1] // 2 : CROP_Y + CROP_SIZE[1] // 2,
        CROP_X - CROP_SIZE[0] // 2 : CROP_X + CROP_SIZE[0] // 2,
    ]

    logger.debug(f"data {ELEVATION_FILE} min: {np.min(data)} | max: {np.max(data)} | shape: {data.shape}")

    # kernel5 = np.ones((5, 5), np.uint8)
    # two_tone_data = cv2.imread(TWO_TONE_FILE, cv2.IMREAD_ANYCOLOR)
    # two_tone_data = cv2.cvtColor(two_tone_data, cv2.COLOR_BGR2HSV)
    # # BLUR_HIGHLIGHT_KERNEL_SIZE = 5
    # # two_tone_data = cv2.blur(two_tone_data, (BLUR_HIGHLIGHT_KERNEL_SIZE, BLUR_HIGHLIGHT_KERNEL_SIZE))
    # two_tone_highlights = cv2.inRange(two_tone_data, np.array([60 - 50, 10, 10]), np.array([60 + 50, 255, 255]))
    # # two_tone_highlights = cv2.morphologyEx(two_tone_highlights, cv2.MORPH_OPEN, kernel5)
    # # two_tone_highlights = cv2.morphologyEx(two_tone_highlights, cv2.MORPH_CLOSE, kernel5)
    # two_tone_highlights = cv2.resize(two_tone_highlights, RESIZE_SIZE, cv2.INTER_NEAREST)
    #
    # two_tone_highlights = two_tone_highlights[
    #     CROP_Y - CROP_SIZE[1] // 2 : CROP_Y + CROP_SIZE[1] // 2,
    #     CROP_X - CROP_SIZE[0] // 2 : CROP_X + CROP_SIZE[0] // 2,
    # ]
    #
    # cv2.imwrite(Path(OUTPUT_PATH, "two_tone_split.png"), two_tone_highlights)

    for i in range(1, 30):
        _, _, _, _, angles, inclination = get_slope(data, i)

        angle_width = 30
        tanako_ang = cv2.inRange(np.degrees(angles), np.array([45 - angle_width / 2]), np.array([45 + angle_width / 2]))
        # tanako_inc = cv2.inRange(inclination, np.array([500]), np.array([np.max(inclination)]))
        tanako_inc = cv2.inRange(inclination, np.array([20]), np.array([2000]))

        tanako = (np.logical_and(tanako_ang, tanako_inc) * 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        tanako = cv2.morphologyEx(tanako, cv2.MORPH_OPEN, kernel)
        tanako = cv2.morphologyEx(tanako, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(Path(OUTPUT_PATH, f"tanako_base_stride_{i}.png"), tanako)
    exit()

    two_tone_highlights = tanako

    # density_data = None
    #
    # if DENSITY_FILE == ELEVATION_FILE:
    #     density_data = np.copy(data)
    # else:
    #     if DENSITY_FILE.suffix.endswith(".tif"):
    #         density_data = cv2.imread(str(ELEVATION_FILE), cv2.IMREAD_UNCHANGED)
    #     else:
    #         density_data = cv2.imread(str(DENSITY_FILE), cv2.IMREAD_GRAYSCALE)
    #
    # if density_data.shape != data.shape:
    #     density_data = cv2.resize(density_data, data.shape)
    #
    # density_normalized = (density_data - np.min(density_data)) / (np.max(density_data) - np.min(density_data))
    #
    # # Apply a non-linear scale
    # scale = scales.Scale(scales.quadratic_bezier, {"p1": [0.30, 0], "p2": [.70, 1.0]})
    # density_normalized = scale.apply(density_normalized)
    #
    # density = (np.full(density_data.shape, config.LINE_DISTANCE[0], dtype=float) +
    #            (density_normalized * (config.LINE_DISTANCE[1] - config.LINE_DISTANCE[0])))

    timer = datetime.datetime.now()

    # pr = profile.Profile()
    # pr.enable()

    tiler = FlowlineTiler(data, None, config, [2, 2])

    # tiler = FlowlineTilerPoly(
    #     data,
    #     None,
    #     config,
    #     [
    #         Point([data.shape[1]*0.25, data.shape[0]//2]).buffer(np.min(data.shape)//4-100),
    #         Point([data.shape[1]*0.75, data.shape[0]//2]).buffer(np.min(data.shape)//4-100)
    #     ]
    # )

    linestrings = tiler.hatch()

    # split linestrings
    linestrings_splitted = []
    for l in linestrings:
        coords = l.coords
        for i in range(len(coords) - 1):
            linestrings_splitted.append(LineString([coords[i], coords[i + 1]]))

    linestrings_lowlights = []
    linestrings_highlights = []
    for l in linestrings_splitted:
        match = False
        for p in l.coords:
            x = int(p[0])
            y = int(p[1])
            if two_tone_highlights[y, x] > 0:
                match = True
                break
        if match:
            linestrings_highlights.append(l)
        else:
            linestrings_lowlights.append(l)

    print(f"high: {len(linestrings_highlights)} / low: {len(linestrings_lowlights)}")

    # pr.disable()
    # pr.dump_stats("profile.pstat")

    total_time = (datetime.datetime.now() - timer).total_seconds()
    avg_line_length = sum([x.length for x in linestrings]) / len(linestrings)

    logger.info(f"total time:         {total_time:5.2f}s")
    logger.info(f"avg line length:    {avg_line_length:5.2f}")

    timer = datetime.datetime.now()
    tiler._debug_viz(linestrings)
    total_time = (datetime.datetime.now() - timer).total_seconds()
    logger.info(f"total time viz:     {total_time:5.2f}s")

    svg = SvgWriter(Path(OUTPUT_PATH, "flowlines.svg"), data.shape)

    options = {"fill": "none", "stroke": "black", "stroke-width": "2"}

    # svg.add("flowlines", linestrings, options=options)

    # land_polys = _extract_polygons(data, *get_elevation_bounds([0, 10_000], 1)[0], True)
    # options_land = {
    #     "fill": "green",
    #     "stroke": "none",
    #     "fill-opacity": "0.5"
    # }
    # svg.add("land", land_polys, options=options_land)

    options_high = {
        "fill": "none",
        "stroke": "aqua",
        # "opacity": "0.5",
        "stroke-width": "2",
    }
    options_low = {"fill": "none", "stroke": "blue", "stroke-width": "2"}

    svg.add("flowlines_high", linestrings_highlights, options=options_high)
    svg.add("flowlines_low", linestrings_lowlights, options=options_low)

    svg.write()
