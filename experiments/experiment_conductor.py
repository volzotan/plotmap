import datetime
import os
import re
import subprocess
import toml
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

import lineworld

# SCRIPT_PATH = "experiments/hatching/flowlines.py"
# WORKING_DIR = "."
# VARIABLE_NAME = "BLUR_ANGLES_KERNEL_SIZE"
# SCRIPT_OUTPUT_IMAGE_PATH = "experiments/hatching/output/flowlines.png"
# OUTPUT_DIR = "experiments/conductor"

SCRIPT_PATH = "lineworld/run.py"
WORKING_DIR = "."
TEMP_DIR = "tmp"
TMP_CONFIG_FILE = Path(TEMP_DIR, "config_overwrite.toml")
VARIABLE = ["layer", "bathymetryflowlines", "blur_angles_kernel_size"]
SCRIPT_OUTPUT_IMAGE_PATH = "test.svg"
OUTPUT_DIR = "experiments/conductor"

FFMPEG_TEMP_FILE = Path(OUTPUT_DIR, "ffmpeg_mux_file.txt")
FFMPEG_OUTPUT_FILE = Path(OUTPUT_DIR, f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4")
FFMPEG_DURATION = 2

VARIABLE_STATES = [5, 9, 13, 17, 21, 31, 41, 51, 61, 81, 101]

INKSCAPE_CONVERSION_SUFFIX = ".png"
INKSCAPE_CONVERSION_WIDTH = 8000
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
    if isinstance(VARIABLE, list):
        tmp_config = config
        for i in range(len(VARIABLE)):
            if i == len(VARIABLE) - 1:
                tmp_config[VARIABLE[i]] = variable_state
            else:
                tmp_config[VARIABLE[i]] = {}
                tmp_config = tmp_config[VARIABLE[i]]

    else:
        config[VARIABLE] = variable_state
    with open(TMP_CONFIG_FILE, 'w') as f:
        toml.dump(config, f)

    # run the script from the correct working dir
    modified_env = os.environ.copy()
    modified_env[lineworld.ENV_OVERWRITE_CONFIG] = TMP_CONFIG_FILE
    result = subprocess.run(
        ["python", SCRIPT_PATH],
        cwd=WORKING_DIR,
        env=modified_env,
        capture_output=False
    )

    if result.returncode != 0:
        raise Exception(f"non-zero return code. Experiment: {variable_name}={variable_state}")

    runtime = (datetime.datetime.now() - timer_start).total_seconds()
    total_runtime += runtime

    # if SVG, convert to image
    experiment_output_image_path = SCRIPT_OUTPUT_IMAGE_PATH
    if SCRIPT_OUTPUT_IMAGE_PATH.lower().endswith(".svg"):
        converted_image_output_path = Path(Path(SCRIPT_OUTPUT_IMAGE_PATH).parent, Path(SCRIPT_OUTPUT_IMAGE_PATH).stem + INKSCAPE_CONVERSION_SUFFIX)
        result = subprocess.run([
            "/Applications/Inkscape.app/Contents/MacOS/inkscape",
            SCRIPT_OUTPUT_IMAGE_PATH,
            f"--export-filename={converted_image_output_path}",
            f"--export-width={INKSCAPE_CONVERSION_WIDTH}"
        ],
            cwd=WORKING_DIR,
            check=True,
            capture_output=False
        )

        experiment_output_image_path = converted_image_output_path

    # insert info text into output image
    img = cv2.imread(str(experiment_output_image_path), cv2.IMREAD_COLOR)
    img_annotated = np.zeros([img.shape[0] + 100, img.shape[1], 3], dtype=np.uint8)
    img_annotated[0:img.shape[0], 0:img.shape[1], :] = img

    cv2.putText(
        img_annotated,
        f"{variable_name}: {str(variable_state):<20}",
        (10, img.shape[0] + 40),
        FONT_NAME, FONT_SCALE,(255, 255, 255), FONT_THICKNESS
    )

    cv2.putText(
        img_annotated,
        f"{datetime.datetime.now().strftime("%Y %m %d | %H:%M:%S")}",
        (10, img.shape[0] + 80),
        FONT_NAME, FONT_SCALE,(255, 255, 255), FONT_THICKNESS
    )

    output_filename = f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_VAR_{variable_name}_{variable_state}{experiment_output_image_path.suffix}"
    img_annotated_path = Path(OUTPUT_DIR, output_filename)
    cv2.imwrite(str(img_annotated_path), img_annotated)

    # print runtime for each script run
    logger.info(f"finished variable: {variable_state:10} | total time: {runtime:>6.2f}s")

    result_dict = {
        "image": img_annotated_path,
        "total_time": runtime
    }
    result_dict["variables"] = {}
    result_dict["variables"][variable_name] = variable_state
    overview.append(result_dict)

logger.info(f"total experiment runtime: {total_runtime:>6.2f}s")

# create a slideshow video file with ffmpeg
# write a concat demuxer file

with open(FFMPEG_TEMP_FILE, "w") as file:
    for res in overview:
        file.write(f"file '{res["image"].name}'\n")
        file.write(f"duration {FFMPEG_DURATION}\n")
    file.write(f"file '{overview[-1]["image"].name}'\n")

#ffmpeg -f concat -i input.txt -vsync vfr -pix_fmt yuv420p output.mp4

result = subprocess.run(
    ["ffmpeg", "-y", "-f", "concat", "-i", str(FFMPEG_TEMP_FILE), "-vsync", "vfr", "-pix_fmt", "yuv420p", str(FFMPEG_OUTPUT_FILE)],
    cwd=WORKING_DIR,
    capture_output=True,
    check=True
)