import math
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import shapely.affinity
import svgpathtools
from loguru import logger
from shapely import LineString, Point
from svgpathtools import parse_path

from lineworld.util.geometrytools import _linestring_to_coordinate_pairs


class Align(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    CENTER = "CENTER"


class HersheyFont:
    DEFAULT_FONT = "fonts/HersheySans1.svg"

    def __init__(self, font_file: Path = DEFAULT_FONT):
        self.font_file = Path(font_file)
        self.parse_svg_font(self.font_file)

    def parse_svg_font(self, filename: Path) -> None:
        namespaces = {"svg": "http://www.w3.org/2000/svg"}
        tree = ET.parse(filename)

        self.font_dict = {}
        self.font_info = {}

        font_face = tree.find(".//svg:font-face", namespaces=namespaces)
        glyphs = tree.findall(".//svg:glyph", namespaces=namespaces)

        scaler = 1 / float(font_face.get("units-per-em", "1000"))
        self.font_info["ascent"] = float(font_face.get("ascent", "800")) * scaler
        self.font_info["descent"] = float(font_face.get("descent", "-200")) * scaler
        self.font_info["cap-height"] = float(font_face.get("cap-height", "500")) * scaler
        self.font_info["x-height"] = float(font_face.get("x-height", "300")) * scaler

        for glyph in glyphs:
            char = glyph.get("unicode")
            self.font_dict[char] = {}
            self.font_dict[char]["glyph-name"] = glyph.get("glyph-name")
            self.font_dict[char]["horiz-adv-x"] = float(glyph.get("horiz-adv-x")) * scaler
            self.font_dict[char]["path"] = glyph.get("d")
            self.font_dict[char]["lines"] = []

            if glyph.get("d") is not None:
                for path in parse_path(glyph.get("d")).continuous_subpaths():
                    linestring_points = []

                    if len(path) == 1:  # path consisting of a single line/curve
                        e = path[0]
                        if type(e) in [svgpathtools.Line, svgpathtools.CubicBezier]:
                            linestring_points.append([e.start.real, e.start.imag])
                            linestring_points.append([e.end.real, e.end.imag])
                        else:
                            logger.warning(f"unknown SVG path element: {e}")
                    else:
                        for i, e in enumerate(path):
                            if type(e) in [svgpathtools.Line, svgpathtools.CubicBezier]:
                                if not i == len(path) - 1:
                                    linestring_points.append([e.start.real, e.start.imag])
                                else:
                                    linestring_points.append([e.start.real, e.start.imag])
                                    linestring_points.append([e.end.real, e.end.imag])
                            else:
                                logger.warning(f"unknown SVG path element: {e}")

                    if len(linestring_points) > 1:
                        ls = LineString(linestring_points)
                        ls = shapely.affinity.scale(ls, xfact=scaler, yfact=-scaler, origin=(0, 0, 0))
                        self.font_dict[char]["lines"].append(ls)

    def glyphs_for_text(self, text: str, font_size: float) -> list[dict[str, Any]]:
        horizontal_advance = 0
        output = []

        for c in text:
            if c not in self.font_dict:
                logger.warning(f"character {c} not part of SVG font {self.font_file}")
                continue

            output_dict = {}

            glyph = self.font_dict[c]
            linestrings = glyph["lines"]
            linestrings = [
                shapely.affinity.scale(ls, xfact=font_size, yfact=font_size, origin=(0, 0, 0)) for ls in linestrings
            ]

            output_dict["char"] = c
            output_dict["lines"] = linestrings
            output_dict["anchor"] = (horizontal_advance, 0)
            output_dict["width"] = glyph["horiz-adv-x"] * font_size

            horizontal_advance += output_dict["width"]
            output.append(output_dict)

        return output

    def _find_matching_line_point(
        self, line: LineString, start: float, end: float, text_length: float, align: Align, reverse_path: bool = False
    ) -> tuple[np.ndarray, float] | None:
        """
        For a given LineString and a start/end point find the closest matching point on the line for the start point
        and the angle to the end point
        """
        coords = line.coords
        if reverse_path:
            coords = list(reversed(coords))

        distance_on_line = np.zeros(len(coords), dtype=float)
        for i in range(1, len(coords)):
            distance_on_line[i] = distance_on_line[i - 1] + math.sqrt(
                (coords[i][0] - coords[i - 1][0]) ** 2 + (coords[i][1] - coords[i - 1][1]) ** 2
            )

        if distance_on_line.shape[0] < 10:
            logger.warning("line has less than 10 segments, probably missing segmentation")

        match_start = None
        match_end = None

        match align:
            case Align.LEFT:
                match_start = coords[int(np.argsort(np.abs(distance_on_line - start))[0])]
                match_end = coords[int(np.argsort(np.abs(distance_on_line - end))[0])]
            case Align.CENTER:
                offset = distance_on_line[-1] / 2 - text_length / 2
                match_start = coords[int(np.argsort(np.abs(distance_on_line - start - offset))[0])]
                match_end = coords[int(np.argsort(np.abs(distance_on_line - end - offset))[0])]
            case Align.RIGHT:
                match_start = coords[int(np.argsort(np.abs(distance_on_line - distance_on_line[-1] + end))[0])]
                match_end = coords[int(np.argsort(np.abs(distance_on_line - distance_on_line[-1] + start))[0])]
            case _:
                raise Exception(f"unknown alignment state {align}")

        if match_start is None or match_end is None:
            return None

        if match_start[0] == match_end[0] and match_start[1] == match_end[1]:
            return None

        angle = math.atan2(match_end[1] - match_start[1], match_end[0] - match_start[0])
        return (match_start, angle)

    def lines_for_text(
        self,
        text: str,
        font_size: float,
        path: LineString = None,
        align: Align = Align.LEFT,
        center_vertical: bool = False,
        reverse_path: bool = False,
    ) -> list[LineString]:
        output = []

        if align == Align.RIGHT:
            text = reversed(text)

        glyphs = self.glyphs_for_text(text, font_size)
        total_text_length = np.sum([g["width"] for g in glyphs])

        for i, g in enumerate(glyphs):
            anchor_x = g["anchor"][0]

            if path is not None:
                match = self._find_matching_line_point(
                    path, anchor_x, anchor_x + g["width"], total_text_length, align=align, reverse_path=reverse_path
                )

                if match is None:
                    logger.warning(f"path length insufficient, failed to draw glyph {g['char']}")
                    continue

                matching_point, angle = match

                for linestring in g["lines"]:
                    linestring = shapely.affinity.rotate(linestring, angle, origin=(0, 0, 0), use_radians=True)
                    linestring = shapely.affinity.translate(linestring, xoff=matching_point[0], yoff=matching_point[1])
                    output.append(linestring)
            else:
                for linestring in g["lines"]:
                    linestring = shapely.affinity.translate(linestring, xoff=g["anchor"][0])
                    output.append(linestring)


        if align == Align.LEFT:
            xoff = glyphs[0]["width"] * -0.3
            output = [shapely.affinity.translate(ls, xoff=xoff) for ls in output]

        if center_vertical:
            yoff = (self.font_info["cap-height"] * font_size) / 2
            output = [shapely.affinity.translate(ls, yoff=yoff) for ls in output]

        return output

    def lines_for_text_along_path_debug(self, text: str, path: LineString, font_size: float, img) -> list[LineString]:
        output = []
        glyphs = self.glyphs_for_text(text, font_size)
        for i, g in enumerate(glyphs):
            anchor_x = g["anchor"][0]  # + g["width"]/2
            match = self._find_matching_line_point(path, anchor_x, anchor_x + g["width"], reverse=True)

            if match is None:
                logger.warning(f"path length insufficient, failed to draw glyph {g['char']}")
                continue

            matching_point, angle = match

            # anchor
            cv2.circle(
                img,
                [int(matching_point[0]), int(matching_point[1])],
                4,
                (0, 0, 255),
                -1,
            )

            # bounding box
            bbox = shapely.box(
                0,
                -self.font_info["descent"] * FONT_SIZE,
                g["width"],
                -self.font_info["ascent"] * FONT_SIZE,
            )
            bbox = shapely.affinity.rotate(bbox, angle, origin=(0, 0, 0), use_radians=True)
            bbox = shapely.affinity.translate(bbox, xoff=matching_point[0], yoff=matching_point[1])

            bbox_coords = np.array(bbox.exterior.coords, dtype=np.int32)
            bbox_coords = bbox_coords.reshape((-1, 1, 2))
            cv2.polylines(img, [bbox_coords], True, (0, 255, 0))

            # character baseline
            char_baseline = LineString([[0, 0], [g["width"], 0]])
            char_baseline = shapely.affinity.rotate(char_baseline, angle, origin=(0, 0, 0), use_radians=True)
            char_baseline = shapely.affinity.translate(char_baseline, xoff=matching_point[0], yoff=matching_point[1])
            for pair in _linestring_to_coordinate_pairs(char_baseline):
                pt1 = [int(c) for c in pair[0]]
                pt2 = [int(c) for c in pair[1]]
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)

            # char
            for linestring in g["lines"]:
                linestring = shapely.affinity.rotate(linestring, angle, origin=(0, 0, 0), use_radians=True)
                linestring = shapely.affinity.translate(linestring, xoff=matching_point[0], yoff=matching_point[1])
                output.append(linestring)

                for pair in _linestring_to_coordinate_pairs(linestring):
                    pt1 = [int(c) for c in pair[0]]
                    pt2 = [int(c) for c in pair[1]]
                    cv2.line(img, pt1, pt2, (0, 0, 0), 4)

        return output


if __name__ == "__main__":
    TEXT = "The quick brown fox jumps over the lazy dog"

    FONT_SIZE = 54

    CANVAS_DIMENSIONS = [1200, 1200]
    OFFSET = [100, CANVAS_DIMENSIONS[1] // 2]

    font = HersheyFont(font_file=Path(Path("../.."), Path(HersheyFont.DEFAULT_FONT)))

    img = np.full(CANVAS_DIMENSIONS + [3], 255, dtype=np.uint8)

    path1 = LineString([[1000, 800], [200, 700]])

    path2 = shapely.intersection(
        LineString(
            list(
                Point([CANVAS_DIMENSIONS[0] / 2], [CANVAS_DIMENSIONS[1] * 0.7])
                .buffer(CANVAS_DIMENSIONS[0] * 0.6)
                .exterior.coords
            )
        ),
        shapely.box(100, 100, CANVAS_DIMENSIONS[0] - 100, CANVAS_DIMENSIONS[1] - 100),
    )
    path2 = path2.segmentize(1)
    linestrings_along_path2 = font.lines_for_text(TEXT, FONT_SIZE, path=path2, align=Align.LEFT, reverse_path=True)

    for linestring in linestrings_along_path2:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(c) for c in pair[0]]
            pt2 = [int(c) for c in pair[1]]
            cv2.line(img, pt1, pt2, (0, 0, 0), 4)

    path3 = shapely.affinity.translate(path2, yoff=200)
    linestrings_along_path3 = font.lines_for_text(TEXT, FONT_SIZE, path=path3, align=Align.RIGHT, reverse_path=True)

    for linestring in linestrings_along_path3:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(c) for c in pair[0]]
            pt2 = [int(c) for c in pair[1]]
            cv2.line(img, pt1, pt2, (0, 0, 0), 4)

    path4 = shapely.affinity.translate(path3, yoff=200)
    linestrings_along_path4 = font.lines_for_text(
        "quick brown fox", FONT_SIZE, path=path4, align=Align.CENTER, reverse_path=True
    )

    for linestring in linestrings_along_path4:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(c) for c in pair[0]]
            pt2 = [int(c) for c in pair[1]]
            cv2.line(img, pt1, pt2, (0, 0, 0), 4)

    path5 = shapely.affinity.translate(path1.segmentize(1), yoff=300)
    linestrings_along_path5 = font.lines_for_text(
        "m QUICK brown fox jumps", FONT_SIZE, path=path5, align=Align.CENTER, center_vertical=True, reverse_path=True
    )

    for pair in _linestring_to_coordinate_pairs(path5):
        pt1 = [int(c) for c in pair[0]]
        pt2 = [int(c) for c in pair[1]]
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)

    for linestring in linestrings_along_path5:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(c) for c in pair[0]]
            pt2 = [int(c) for c in pair[1]]
            cv2.line(img, pt1, pt2, (0, 0, 0), 4)




    linestrings = [
        shapely.affinity.translate(l, yoff=+CANVAS_DIMENSIONS[1] * 0.75) for l in font.lines_for_text(TEXT, FONT_SIZE)
    ]

    for linestring in linestrings:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(c) for c in pair[0]]
            pt2 = [int(c) for c in pair[1]]
            cv2.line(img, pt1, pt2, (0, 0, 0), 4)

    for pair in _linestring_to_coordinate_pairs(path2):
        pt1 = [int(c) for c in pair[0]]
        pt2 = [int(c) for c in pair[1]]
        cv2.line(img, pt1, pt2, (0, 0, 0), 2)

    for pair in _linestring_to_coordinate_pairs(path2):
        pt1 = [int(c) for c in pair[0]]
        pt2 = [int(c) for c in pair[1]]
        cv2.circle(img, pt2, 1, (0, 0, 255), -1)

    cv2.imwrite("../../output.png", img)
