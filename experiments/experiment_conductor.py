import datetime
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

SCRIPT_PATH = "experiments/hatching/flowlines.py"
WORKING_DIR = "."
VARIABLE_NAME = "BLUR_ANGLES_KERNEL_SIZE"
SCRIPT_OUTPUT_IMAGE_PATH = "experiments/hatching/output/flowlines.png"
OUTPUT_DIR = "experiments/conductor"

VARIABLE_STATES = [1, 5, 9, 13, 17, 21, 31, 41, 51, 61, 71, 81, 91, 101]

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

    # run the script from the correct working dir
    result = subprocess.run(
        ["python", SCRIPT_PATH, f"--{VARIABLE_NAME}={variable_state}"],
        cwd=WORKING_DIR,
        capture_output=True
    )

    if result.returncode != 0:
        raise Exception(f"non-zero return code. Experiment: {VARIABLE_NAME}={variable_state}")

    runtime = (datetime.datetime.now() - timer_start).total_seconds()
    total_runtime += runtime

    # insert info text into output image
    img = cv2.imread(SCRIPT_OUTPUT_IMAGE_PATH, cv2.IMREAD_COLOR)
    img_annotated = np.zeros([img.shape[0] + 100, img.shape[1], 3], dtype=np.uint8)
    img_annotated[0:img.shape[0], 0:img.shape[1], :] = img

    cv2.putText(
        img_annotated,
        f"{VARIABLE_NAME}: {str(variable_state):<20}",
        (10, img.shape[0] + 40),
        FONT_NAME, FONT_SCALE,(255, 255, 255), FONT_THICKNESS
    )

    cv2.putText(
        img_annotated,
        f"{datetime.datetime.now().strftime("%Y %m %d | %H:%M:%S")}",
        (10, img.shape[0] + 80),
        FONT_NAME, FONT_SCALE,(255, 255, 255), FONT_THICKNESS
    )

    output_filename = f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_VAR_{VARIABLE_NAME}_{variable_state}{Path(SCRIPT_OUTPUT_IMAGE_PATH).suffix}"
    img_annotated_path = Path(OUTPUT_DIR, output_filename)
    cv2.imwrite(str(img_annotated_path), img_annotated)

    # print runtime for each script run
    logger.info(f"finished variable: {variable_state:10} | total time: {runtime:>6.2f}s")

    result_dict = {
        "image": img_annotated_path,
        "total_time": runtime
    }
    result_dict["variables"] = {}
    result_dict["variables"][VARIABLE_NAME] = variable_state
    overview.append(result_dict)

logger.info(f"total experiment runtime: {total_runtime:>6.2f}s")

# create a slideshow video file with ffmpeg
# write a concat demuxer file
FFMPEG_TEMP_FILE = Path(OUTPUT_DIR, "ffmpeg_mux_file.txt")
FFMPEG_OUTPUT_FILE = Path(OUTPUT_DIR, f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4")
FFMPEG_DURATION = 3

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