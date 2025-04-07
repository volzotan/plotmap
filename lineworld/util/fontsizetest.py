from pathlib import Path

import shapely.affinity

from lineworld.core.svgwriter import SvgWriter
from lineworld.util.hersheyfont import HersheyFont

TEXT = ["the quick brown fox", "jumps over the lazy hedgehog"]

FONT_SIZES = [3, 4, 5, 6, 8, 10]
CANVAS_DIMENSIONS = [210 - 20, 297 - 20]
OUTPUT_PATH = "."

linestrings = []
offset = 10
for font_size in FONT_SIZES:
    font = HersheyFont(font_file=Path(HersheyFont.DEFAULT_FONT))

    lines = font.lines_for_text(f"SIZE: {font_size:5.1f}", font_size)
    linestrings += [shapely.affinity.translate(l, xoff=0, yoff=offset) for l in lines]
    offset += font_size

    for segment in TEXT:
        lines = font.lines_for_text(segment, font_size)
        linestrings += [shapely.affinity.translate(l, xoff=0, yoff=offset) for l in lines]
        offset += font_size + 1

    for segment in TEXT:
        lines = font.lines_for_text(segment.upper(), font_size)
        linestrings += [shapely.affinity.translate(l, xoff=0, yoff=offset) for l in lines]
        offset += font_size + 1

    offset += font_size + 2

svg_path = Path(OUTPUT_PATH, "fontsizetest.svg")
svg = SvgWriter(svg_path, CANVAS_DIMENSIONS)
options = {"fill": "none", "stroke": "black", "stroke-width": "0.5"}
svg.add("lines", linestrings, options=options)
svg.write()
