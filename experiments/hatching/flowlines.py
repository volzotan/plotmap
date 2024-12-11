import datetime
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
import shapely
from loguru import logger
from matplotlib import pyplot as plt
from shapely import LineString, Polygon

from experiments.hatching.slope import get_slope
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.gebco_grid_to_polygon import _extract_polygons, get_elevation_bounds


@dataclass
class FlowlineHatcherConfig():
    LINE_DISTANCE: tuple[float, float] = (2, 80)  # distance between lines
    LINE_STEP_DISTANCE: float = 1.0  # distance between points constituting a line

    MAX_ANGLE_DISCONTINUITY: float = math.pi / 4  # max difference (in radians) in slope between line points
    MIN_INCLINATION: float = 0.1  # 50.0

    SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: int = 20  # How many line segments should be skipped before the next seedpoint is extracted
    LINE_MAX_SEGMENTS: int = 300

    BLUR_ANGLES: bool = True
    BLUR_ANGLES_KERNEL_SIZE: int = 11
    BLUR_DENSITY_MAP: bool = False

    COLLISION_APPROXIMATE: bool = True

    VIZ_LINE_THICKNESS: int = 5

    SIMPLIFY_TOLERANCE: float = 1.0


class FlowlineTiler():

    def __init__(self,
                 elevation: np.ndarray,
                 density: np.ndarray,
                 config: FlowlineHatcherConfig,
                 num_tiles: tuple[int, int]):

        self.elevation = elevation
        self.density = density
        self.config = config
        self.num_tiles = num_tiles

        self.tiles: list[list[dict[str, int | np.ndarray]]] = \
            [[{} for _ in range(num_tiles[0])] for _ in range(num_tiles[1])]

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

                # TODO: extract first_stage_points from the point_raster of already processed neighbouring tiles

                first_stage_points = []

                if col > 0:

                    point_raster_left = self.tiles[col-1][row]["raster"]
                    for y in point_raster_left[:, -2].nonzero()[0]:
                        first_stage_points.append([0, y])

                if row > 0:

                    point_raster_top = self.tiles[col][row-1]["raster"]
                    for x in point_raster_top[-2, :].nonzero()[0]:
                        first_stage_points.append([x, 0])

                print(f"first stage points: {len(first_stage_points)}")

                hatcher = FlowlineHatcher(
                    shapely.box(0, 0, self.col_size, self.row_size),
                    self.elevation[t["min_row"]:t["max_row"], t["min_col"]:t["max_col"]],
                    self.density[t["min_row"]:t["max_row"], t["min_col"]:t["max_col"]],
                    self.config,
                    first_stage_points=first_stage_points
                )

                linestrings = hatcher.hatch()
                linestrings = [shapely.affinity.translate(ls, xoff=t["min_col"], yoff=t["min_row"]) for ls in
                               linestrings]

                self.tiles[row][col]["linestrings"] = linestrings
                self.tiles[row][col]["raster"] = hatcher.point_raster

        linestrings = []
        for col in range(self.num_tiles[0]):
            for row in range(self.num_tiles[1]):
                linestrings += self.tiles[row][col]["linestrings"]

        return linestrings


class FlowlineHatcher():

    def __init__(self, polygon: Polygon,
                 elevation: np.ndarray,
                 density: np.ndarray,
                 config: FlowlineHatcherConfig,
                 first_stage_points=[]):

        self.polygon = polygon
        self.config = config

        _, _, _, _, angles, inclination = get_slope(elevation, 1)

        self.elevation = elevation # np.pad(elevation, (1, 1), "edge")[1:, 1:]
        self.angles = angles # np.pad(angles, (1, 1), "edge")[1:, 1:]
        self.inclination = inclination # np.pad(inclination, (1, 1), "edge")[1:, 1:]
        self.density = density # np.pad(density, (1, 1), "edge")[1:, 1:]

        self.bbox = self.polygon.bounds
        self.bbox = [0,
                     0,
                     math.ceil(self.bbox[2] - self.bbox[0]),
                     math.ceil(self.bbox[3] - self.bbox[1])]  # minx, miny, maxx, maxy

        if self.config.BLUR_ANGLES:
            self.angles = cv2.blur(
                self.angles,
                (self.config.BLUR_ANGLES_KERNEL_SIZE, self.config.BLUR_ANGLES_KERNEL_SIZE)
            )
            # self.angles = cv2.GaussianBlur(self.angles, (11, 11), 0)

        if self.config.BLUR_DENSITY_MAP:
            self.density = cv2.blur(self.density, (60, 60))

        if self.config.COLLISION_APPROXIMATE:
            self.point_raster = np.zeros([self.bbox[3] + 1, self.bbox[2] + 1], dtype=bool)
        else:
            self.point_map = {}
            for x in range(self.bbox[2] + 1):
                for y in range(self.bbox[3] + 1):
                    self.point_map[f"{x},{y}"] = []

        self.first_stage_points = first_stage_points

    def _collision_approximate(self, x: float, y: float) -> bool:
        x = int(x)
        y = int(y)
        half_d = int(self.density[y, x] / 2)

        return np.any(
            self.point_raster[
            max(y - half_d, 0):min(y + half_d, self.point_raster.shape[0]),
            max(x - half_d, 0):min(x + half_d, self.point_raster.shape[1])
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

        output = np.full([self.elevation.shape[0], self.elevation.shape[1], 3], 255, dtype=np.uint8)
        for ls in linestrings:
            line_points = ls.coords
            for i in range(len(line_points) - 1):
                start = (round(line_points[i][0]), round(line_points[i][1]))
                end = (round(line_points[i + 1][0]), round(line_points[i + 1][1]))
                cv2.line(output, start, end, (0, 0, 0), self.config.VIZ_LINE_THICKNESS)
        output[self.point_raster, :] = [0, 0, 255]
        cv2.imwrite(Path(OUTPUT_PATH, "flowlines.png"), output)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

        ax1.imshow(self.elevation, cmap="binary")
        ax1.set_title("elevation")

        ax2.imshow(self.angles, cmap="hsv")
        ax2.set_title("angles")

        ax3.imshow(self.inclination)
        ax3.set_title("inclination")

        ax4.imshow(self.density, cmap="gray")
        ax4.set_title("density")

        ax5.imshow(self.point_raster)
        ax5.set_title("collision map")

        ax6.imshow(output)
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

        starting_points_priority = deque(self.first_stage_points)

        while len(starting_points) > 0:

            seed = None
            if len(starting_points_priority) > 0:
                seed = starting_points_priority.popleft()
            else:
                # pass
                seed = starting_points.popleft()

            if seed is None:
                break

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

            linestrings.append(LineString(line_points).simplify(self.config.SIMPLIFY_TOLERANCE))

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


def _extract_first_stage_points(img: np.ndarray) -> list[tuple[float, float]]:
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 1

    params.filterByInertia = False
    params.filterByConvexity = False

    params.minThreshold = 0
    params.maxThreshold = 200

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    # im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite("blobs.png", im_with_keypoints)

    return [k.pt for k in keypoints]  # x,y order


# ELEVATION_FILE = Path("experiments/hatching/data/GebcoToBlender/reproject.tif")
ELEVATION_FILE = Path("experiments/hatching/data/flowlines_gebco_crop.tif")
# FIRST_STAGE_FILE = Path("experiments/hatching/data/flowlines_gebco_crop_ridges.png")
# DENSITY_FILE = Path("shaded_relief4.png")
DENSITY_FILE = ELEVATION_FILE

# ELEVATION_FILE = Path("experiments/hatching/data/slope_test_5.tif")
# DENSITY_FILE = ELEVATION_FILE
# TARGET_RESOLUTION = [1000, 1000]

OUTPUT_PATH = Path("experiments/hatching/output")

if __name__ == "__main__":

    config = FlowlineHatcherConfig()

    # data = cv2.imread(str(ELEVATION_FILE), cv2.IMREAD_UNCHANGED)^
    # first_stage_data = cv2.imread(str(FIRST_STAGE_FILE), cv2.IMREAD_GRAYSCALE)

    data = None
    with rasterio.open(str(ELEVATION_FILE)) as dataset:
        data = dataset.read(out_shape=(10000, 10000), resampling=rasterio.enums.Resampling.bilinear)[0]
        # data = dataset.read(1)

    logger.debug(f"data {ELEVATION_FILE} min: {np.min(data)} | max: {np.max(data)}")

    density_data = None

    if DENSITY_FILE == ELEVATION_FILE:
        density_data = np.copy(data)
    else:
        if DENSITY_FILE.suffix.endswith(".tif"):
            density_data = cv2.imread(str(ELEVATION_FILE), cv2.IMREAD_UNCHANGED)
        else:
            density_data = cv2.imread(str(DENSITY_FILE), cv2.IMREAD_GRAYSCALE)

    # data = cv2.resize(data, [10000, 10000])
    # first_stage_data = cv2.resize(first_stage_data, data.shape)

    if density_data.shape != data.shape:
        density_data = cv2.resize(density_data, data.shape)

    density_normalized = (density_data - np.min(density_data)) / (np.max(density_data) - np.min(density_data))
    density = (np.full(density_data.shape, config.LINE_DISTANCE[0], dtype=float) +
               (density_normalized * (config.LINE_DISTANCE[1] - config.LINE_DISTANCE[0])))

    first_stage_points = []
    # first_stage_points = _extract_first_stage_points(first_stage_data)

    timer = datetime.datetime.now()



    # pr = profile.Profile()
    # pr.enable()

    # hatcher = FlowlineHatcher(
    #     shapely.box(0, 0, data.shape[1], data.shape[0]),
    #     data,
    #     angles,
    #     inclination,
    #     density,
    #     config,
    #     first_stage_points=first_stage_points
    # )
    #
    # # for p in first_stage_points:
    # #     hatcher.point_raster[int(p[1]), int(p[0])] = True
    #
    # linestrings = hatcher.hatch()

    # pr.disable()
    # pr.dump_stats("profile.pstat")

    tiler = FlowlineTiler(
        data,
        density,
        config,
        [2, 1]
    )

    linestrings = tiler.hatch()

    total_time = (datetime.datetime.now() - timer).total_seconds()
    avg_line_length = sum([x.length for x in linestrings]) / len(linestrings)

    logger.info(f"total time:         {total_time:5.2f}s")
    logger.info(f"avg line length:    {avg_line_length:5.2f}")

    # hatcher._debug_viz(linestrings)

    svg = SvgWriter(Path(OUTPUT_PATH, "flowlines.svg"), data.shape)

    options = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "2"
    }
    svg.add("flowlines", linestrings, options=options)

    # land_polys = _extract_polygons(data, *get_elevation_bounds([0, 10_000], 1)[0], True)
    # options_land = {
    #     "fill": "green",
    #     "stroke": "none",
    #     "fill-opacity": "0.5"
    # }
    # svg.add("land", land_polys, options=options_land)

    svg.write()
