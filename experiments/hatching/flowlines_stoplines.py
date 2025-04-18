import cProfile as profile
import datetime
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import shapely
from loguru import logger
from matplotlib import pyplot as plt
from shapely import LineString, Polygon

from experiments.hatching.slope import get_slope
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.gebco_grid_to_polygon import _extract_polygons, get_elevation_bounds


@dataclass
class FlowlineHatcherConfig:
    LINE_DISTANCE: tuple[float, float] = (2, 40)  # distance between lines
    LINE_STEP_DISTANCE: float = 1.0  # distance between points constituting a line

    MAX_ANGLE_DISCONTINUITY: float = math.pi / 2  # max difference (in radians) in slope between line points
    MIN_INCLINATION: float = 0.1  # 50.0

    SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: int = (
        20  # How many line segments should be skipped before the next seedpoint is extracted
    )
    LINE_MAX_SEGMENTS: int = 300

    BLUR_ANGLES: bool = True
    BLUR_DENSITY_MAP: bool = True

    COLLISION_APPROXIMATE: bool = True


class FlowlineHatcher:
    def __init__(
        self,
        polygon: Polygon,
        elevation: np.ndarray,
        angles: np.ndarray,
        inclination: np.ndarray,
        density: np.ndarray,
        config: FlowlineHatcherConfig,
    ):
        self.polygon = polygon
        self.config = config

        self.elevation = np.pad(elevation, (1, 1), "edge")[1:, 1:]
        self.angles = np.pad(angles, (1, 1), "edge")[1:, 1:]
        self.inclination = np.pad(inclination, (1, 1), "edge")[1:, 1:]
        self.density = np.pad(density, (1, 1), "edge")[1:, 1:]

        self.bbox = self.polygon.bounds
        self.bbox = [
            0,
            0,
            math.ceil(self.bbox[2] - self.bbox[0]),
            math.ceil(self.bbox[3] - self.bbox[1]),
        ]  # minx, miny, maxx, maxy

        if self.config.BLUR_ANGLES:
            self.angles = cv2.blur(self.angles, (100, 100))
            # self.angles = cv2.GaussianBlur(self.angles, (11, 11), 0)

        if self.config.BLUR_DENSITY_MAP:
            self.density = cv2.blur(self.density, (60, 60))

        self.point_map = {}
        for x in range(self.bbox[2] + 1):
            for y in range(self.bbox[3] + 1):
                self.point_map[f"{x},{y}"] = []

        self.point_raster = np.zeros([self.bbox[3] + 1, self.bbox[2] + 1], dtype=bool)

    def _collision_approximate(self, x: float, y: float) -> bool:
        x = int(x)
        y = int(y)
        half_d = int(self.density[y, x] / 2)

        return np.any(
            self.point_raster[
                max(y - half_d, 0) : min(y + half_d, self.point_raster.shape[0]),
                max(x - half_d, 0) : min(x + half_d, self.point_raster.shape[1]),
            ]
        )

    def _collision_precise(self, x: float, y: float) -> bool:
        rx = round(x)
        ry = round(y)
        d = self.density[ry, rx]
        half_d = math.ceil(d / 2)

        x_minmax = [max(rx - half_d, 0), min(rx + half_d, self.point_raster.shape[1])]
        y_minmax = [max(ry - half_d, 0), min(ry + half_d, self.point_raster.shape[0])]

        for ix in range(*x_minmax):
            for iy in range(*y_minmax):
                for p in self.point_map[f"{ix},{iy}"]:
                    if math.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2) < d:
                        return True

        return False

    def _collision(self, x: float, y: float) -> bool:
        if self.config.COLLISION_APPROXIMATE:
            return self._collision_approximate(x, y)
        else:
            return self._collision_precise(x, y)

    def _next_point(self, x1: float, y1: float, forwards: bool) -> float:
        rx1 = int(x1)
        ry1 = int(y1)

        a1 = self.angles[ry1, rx1]
        inc = self.inclination[ry1, rx1]

        if abs(inc) < self.config.MIN_INCLINATION:
            return None

        dir = 1
        if not forwards:
            dir = -1

        x2 = x1 + self.config.LINE_STEP_DISTANCE * math.cos(a1) * dir
        y2 = y1 + self.config.LINE_STEP_DISTANCE * math.sin(a1) * dir

        # if not self.polygon.contains(Point(x2, y2)):
        #     return None

        if x2 < 0 or x2 > self.bbox[2] or y2 < 0 or y2 > self.bbox[3]:  # TODO
            return None

        if self._collision(x2, y2):
            return None

        if self.config.MAX_ANGLE_DISCONTINUITY > 0:
            a2 = self.angles[int(y2), int(x2)]

            if abs(a2 - a1) > self.config.MAX_ANGLE_DISCONTINUITY:
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

            x4 = self.density[int(y3), int(x3)]
            y4 = 0

            x5 = x4 * math.cos(a2) - y4 * math.sin(a2) + x3
            y5 = x4 * math.sin(a2) + y4 * math.cos(a2) + y3

            # if not self.polygon.contains(Point([x5, y5])):
            #     continue

            if x5 < 0 or x5 > self.bbox[2] or y5 < 0 or y5 > self.bbox[3]:  # TODO
                continue

            seed_points.append([x5, y5])

        return seed_points

    def _debug_viz(self, linestrings: list[LineString]) -> None:
        output = np.zeros([self.elevation.shape[0], self.elevation.shape[1], 3], dtype=np.uint8)
        for ls in linestrings:
            line_points = ls.coords
            for i in range(len(line_points) - 1):
                start = (round(line_points[i][0]), round(line_points[i][1]))
                end = (round(line_points[i + 1][0]), round(line_points[i + 1][1]))
                cv2.line(output, start, end, (255, 255, 255), 2)
        cv2.imwrite(Path(OUTPUT_PATH, "flowlines.png"), ~output)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

        ax1.imshow(self.elevation, cmap="binary")
        ax1.set_title("elevation")

        ax2.imshow(self.angles)
        ax2.set_title("angles")

        ax3.imshow(self.inclination)
        ax3.set_title("inclination")

        ax4.imshow(self.density, cmap="gray")
        ax4.set_title("density")

        ax5.imshow(self.point_raster)
        ax5.set_title("collision map")

        ax6.imshow(~output)
        ax6.set_title("output")

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

    def hatch(self) -> list[LineString]:
        # output = np.zeros([1000, 1000, 3], dtype=np.uint8)

        linestrings = []
        starting_points = deque()

        for i in np.linspace(self.bbox[0] + 1, self.bbox[2] - 1, num=100):
            for j in np.linspace(self.bbox[1] + 1, self.bbox[3] - 1, num=100):
                starting_points.append([i, j])

        while len(starting_points) > 0:
            seed = starting_points.popleft()

            if self._collision(*seed):
                continue

            line_points = deque([seed])

            # follow gradient up
            for i in range(10000):
                if self.config.LINE_MAX_SEGMENTS > 0 and len(line_points) >= self.config.LINE_MAX_SEGMENTS:
                    break

                p = self._next_point(*line_points[-1], True)

                if p is None:
                    break

                line_points.append(p)

            # follow gradient down
            for i in range(10000):
                if self.config.LINE_MAX_SEGMENTS > 0 and len(line_points) >= self.config.LINE_MAX_SEGMENTS:
                    break

                p = self._next_point(*line_points[0], False)

                if p is None:
                    break

                line_points.appendleft(p)

            if len(line_points) < 2:
                continue

            linestrings.append(LineString(line_points))

            # seed points
            seed_points = self._seed_points(line_points)
            starting_points.extendleft(seed_points)

            # collision checks
            for lp in line_points:
                x = int(lp[0])
                y = int(lp[1])
                if self.config.COLLISION_APPROXIMATE:
                    self.point_raster[y, x] = True
                else:
                    self.point_map[f"{x},{y}"].append(lp)

            # viz
            # cv2.circle(output, (round(seed[0]), round(seed[1])), 2, (255, 0, 0), -1)
            # for i in range(len(line_points) - 1):
            #     start = (round(line_points[i][0]), round(line_points[i][1]))
            #     end = (round(line_points[i + 1][0]), round(line_points[i + 1][1]))
            #     cv2.line(output, start, end, (255, 255, 255), 2)

        return linestrings


ELEVATION_FILE = Path("experiments/hatching/data/flowlines_gebco_crop.tif")
DENSITY_FILE = ELEVATION_FILE

# ELEVATION_FILE = Path("experiments/hatching/data/slope_test_5.tif")
# DENSITY_FILE = ELEVATION_FILE
# TARGET_RESOLUTION = [1000, 1000]

OUTPUT_PATH = Path("experiments/hatching/output")

if __name__ == "__main__":
    # sanity checks:

    # if not (LINE_STEP_DISTANCE < LINE_DISTANCE[0]):
    #     raise Exception("distance between points of a line must be smaller than the distance between lines")

    # self.polygon = shapely.box(0, 0, 999, 999)
    # self.polygon = Point([500, 500]).buffer(450)

    c = FlowlineHatcherConfig()

    data = cv2.imread(str(ELEVATION_FILE), cv2.IMREAD_UNCHANGED)

    logger.debug(f"data {ELEVATION_FILE} min: {np.min(data)} | max: {np.max(data)}")

    density_data = None
    if DENSITY_FILE.suffix.endswith(".tif"):
        density_data = cv2.imread(str(ELEVATION_FILE), cv2.IMREAD_UNCHANGED)
    else:
        density_data = cv2.imread(str(DENSITY_FILE), cv2.IMREAD_GRAYSCALE)

    density_normalized = (density_data - np.min(density_data)) / (np.max(density_data) - np.min(density_data))
    density = np.full(density_data.shape, c.LINE_DISTANCE[0], dtype=float) + (
        density_normalized * (c.LINE_DISTANCE[1] - c.LINE_DISTANCE[0])
    )

    X, Y, dX, dY, angles, inclination = get_slope(data, 10)

    timer = datetime.datetime.now()

    hatcher = FlowlineHatcher(
        shapely.box(0, 0, data.shape[1], data.shape[0]),
        data,
        angles,
        inclination,
        density,
        c,
    )

    pr = profile.Profile()
    pr.enable()

    linestrings = hatcher.hatch()

    pr.disable()
    pr.dump_stats("profile.pstat")

    total_time = (datetime.datetime.now() - timer).total_seconds()
    avg_line_length = sum([x.length for x in linestrings]) / len(linestrings)

    logger.info(f"total time:         {total_time:5.2f}s")
    logger.info(f"avg line length:    {avg_line_length:5.2f}")

    hatcher._debug_viz(linestrings)

    svg = SvgWriter(Path(OUTPUT_PATH, "flowlines.svg"), data.shape)

    options = {"fill": "none", "stroke": "black", "stroke-width": "2"}
    svg.add("flowlines", linestrings, options=options)

    land_polys = _extract_polygons(data, *get_elevation_bounds([0, 10_000], 1)[0], True)
    options_land = {"fill": "green", "stroke": "none", "fill-opacity": "0.5"}
    svg.add("land", land_polys, options=options_land)

    svg.write()
