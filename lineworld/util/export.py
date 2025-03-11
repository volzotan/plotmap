import subprocess
from pathlib import Path

import lineworld
from lineworld.core.svgwriter import SvgWriter


def convert_svg_to_png(
    svgWriter: SvgWriter,
    image_width: int | None = None,
    working_dir: Path = Path("."),
    inkscape_conversion_suffix=".png",
) -> None:
    svg_filename = svgWriter.filename

    if image_width is None:
        image_width = svgWriter.dimensions[0]

    converted_image_output_path = Path(Path(svg_filename).parent, Path(svg_filename).stem + inkscape_conversion_suffix)
    background_color = svgWriter.background_color if svgWriter.background_color is not None else "white"
    inkscape_command = lineworld.get_config().get("inkscape_command", "inkscape")

    result = subprocess.run(
        [
            inkscape_command,
            svg_filename,
            f"--export-filename={converted_image_output_path}",
            f"--export-width={image_width}",
            f"--export-background={background_color}",
        ],
        cwd=working_dir,
        check=True,
        capture_output=False,
    )
