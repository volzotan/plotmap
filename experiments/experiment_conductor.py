import datetime
import os
import re
import subprocess
import tomllib

import toml
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

import lineworld

# VARIABLE = ["layer", "BathymetryFlowlines", "blur_angles_kernel_size"]
# VARIABLE = ["layer", "BathymetryFlowlines", "blur_density_kernel_size"]
# VARIABLE = ["layer", "BathymetryFlowlines", "line_distance"]
# VARIABLE = ["layer", "BathymetryFlowlines", "scale_adjustment_value"]
VARIABLE = ["layer", "BathymetryFlowlines", "line_distance_end_factor"]
# VARIABLE = ["layer", "BathymetryFlowlines", "line_max_segments"]

# VARIABLE_STATES = [1, 3, 5, 9, 13, 17, 21, 31, 41, 51, 61, 81, 101]
# VARIABLE_STATES = [
#     # [0.5, 3.0],
#     # [0.5, 4.0],
#     [0.5, 5.0],
#     # [0.5, 6.0],
#     # [0.5, 7.0],
#     # [0.5, 10.0]
#     # [3, 6],
# ]
VARIABLE_STATES = [0.2, 0.4, 0.6, 0.8, 1.0]
# VARIABLE_STATES = [5, 10, 20, 30, 40, 50]

SCRIPT_PATH = "lineworld/run.py"
WORKING_DIR = "."
TEMP_DIR = "temp"
BASE_CONFIG_FILE = Path("configs", "config_750x500.toml")
TMP_CONFIG_FILE = Path(TEMP_DIR, "config_overwrite.toml")
OUTPUT_DIR = "experiments/conductor"

FFMPEG_TEMP_FILE = Path(OUTPUT_DIR, "ffmpeg_mux_file.txt")
FFMPEG_OUTPUT_FILE = Path(OUTPUT_DIR, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
FFMPEG_DURATION = 1

INKSCAPE_CONVERSION_SUFFIX = ".png"
INKSCAPE_CONVERSION_WIDTH = 20000
FONT_NAME = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2

# ----------------------------------------------------------------------------------------------------------------------

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.debug(f"created OUTPUT_DIR {OUTPUT_DIR}")

total_runtime = 0.0

overview = []

for variable_state in VARIABLE_STATES:
    timer_start = datetime.datetime.now()

    variable_name = ".".join(VARIABLE)

    # create the temporary config overwrite file
    config = {}
    with open(BASE_CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)

    if isinstance(VARIABLE, list):
        tmp_config = config
        for i in range(len(VARIABLE)):
            if i == len(VARIABLE) - 1:
                tmp_config[VARIABLE[i]] = variable_state
            else:
                if VARIABLE[i] not in tmp_config:
                    tmp_config[VARIABLE[i]] = {}
                tmp_config = tmp_config[VARIABLE[i]]
    else:
        config[VARIABLE] = variable_state

    with open(TMP_CONFIG_FILE, "w") as f:
        toml.dump(config, f)

    variable_state_printable = re.sub("[\[\]]", "", str(variable_state))

    # run the script from the correct working dir
    modified_env = os.environ.copy()
    modified_env[lineworld.ENV_OVERWRITE_CONFIG] = TMP_CONFIG_FILE
    result = subprocess.run(["python", SCRIPT_PATH], cwd=WORKING_DIR, env=modified_env, capture_output=False)

    if result.returncode != 0:
        raise Exception(f"non-zero return code. Experiment: {variable_name}={variable_state}")

    runtime = (datetime.datetime.now() - timer_start).total_seconds()
    total_runtime += runtime

    # if SVG, convert to image
    experiment_output_image_path = config["name"] + ".svg"
    if experiment_output_image_path.lower().endswith(".svg"):
        converted_image_output_path = Path(
            Path(experiment_output_image_path).parent,
            Path(experiment_output_image_path).stem + INKSCAPE_CONVERSION_SUFFIX,
        )
        result = subprocess.run(
            [
                "/Applications/Inkscape.app/Contents/MacOS/inkscape",
                experiment_output_image_path,
                f"--export-filename={converted_image_output_path}",
                f"--export-width={INKSCAPE_CONVERSION_WIDTH}",
            ],
            cwd=WORKING_DIR,
            check=True,
            capture_output=False,
        )

        # os.remove(experiment_output_image_path)
        experiment_output_image_path = converted_image_output_path

    # insert info text into output image
    img = cv2.imread(str(experiment_output_image_path), cv2.IMREAD_COLOR)
    img_annotated = np.zeros([img.shape[0] + 100, img.shape[1], 3], dtype=np.uint8)
    img_annotated[0 : img.shape[0], 0 : img.shape[1], :] = img

    cv2.putText(
        img_annotated,
        f"{variable_name}: {str(variable_state_printable):<20}",
        (10, img.shape[0] + 40),
        FONT_NAME,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
    )

    cv2.putText(
        img_annotated,
        f"{datetime.datetime.now().strftime('%Y %m %d | %H:%M:%S')}",
        (10, img.shape[0] + 80),
        FONT_NAME,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
    )

    output_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_VAR_{variable_name}_{variable_state_printable}{experiment_output_image_path.suffix}"
    img_annotated_path = Path(OUTPUT_DIR, output_filename)
    cv2.imwrite(str(img_annotated_path), img_annotated)

    # print runtime for each script run
    logger.info(f"finished variable: {variable_state_printable:10} | total time: {runtime:>6.2f}s")

    result_dict = {"image": img_annotated_path, "total_time": runtime}
    result_dict["variables"] = {}
    result_dict["variables"][variable_name] = variable_state
    overview.append(result_dict)

logger.info(f"total experiment runtime: {total_runtime:>6.2f}s")

# create a slideshow video file with ffmpeg
# write a concat demuxer file

with open(FFMPEG_TEMP_FILE, "w") as file:
    for res in overview:
        file.write(f"file '{res['image'].name}'\n")
        file.write(f"duration {FFMPEG_DURATION}\n")
    file.write(f"file '{overview[-1]['image'].name}'\n")

# ffmpeg -f concat -i input.txt -vsync vfr -pix_fmt yuv420p output.mp4

result = subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-i",
        str(FFMPEG_TEMP_FILE),
        "-vsync",
        "vfr",
        "-pix_fmt",
        "yuv420p",
        str(FFMPEG_OUTPUT_FILE),
    ],
    cwd=WORKING_DIR,
    capture_output=True,
    check=True,
)
