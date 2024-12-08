import datetime
import os
import re
import subprocess
from pathlib import Path

import cv2
from loguru import logger

TEMP_DIR = "experiments/temp"

SCRIPT_PATH = "experiments/hatching/flowlines_two_stage.py"
WORKING_DIR = "."
VARIABLE_NAME = "BLUR_ANGLES_KERNEL_SIZE"
SCRIPT_OUTPUT_IMAGE_PATH = "experiments/hatching/output/flowlines.png"
OUTPUT_DIR = "experiments/conductor"

VARIABLE_STATES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31]

# ---

# if not os.path.exists(TEMP_DIR):
#     os.makedirs(TEMP_DIR)
#     logger.debug(f"created TEMP_DIR {TEMP_DIR}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.debug(f"created OUTPUT_DIR {OUTPUT_DIR}")

total_runtime = 0.0

for variable_state in VARIABLE_STATES:
    # create temp copy of script
    temp_script_path = Path(Path(SCRIPT_PATH).parent, "temp_experiment_" + Path(SCRIPT_PATH).name)

    # Replace VARIABLE = VALUE in temp script
    p = re.compile(f"({VARIABLE_NAME}(?:: *[a-zA-Z]* *)?= ?)([^\r\n]*)[\r\n]+")
    r = r"\1 " + str(variable_state) + "\n"
    with open(SCRIPT_PATH, 'r') as input_file:
        with open(temp_script_path, 'w') as output_file:
            for line in input_file:
                line = p.sub(r, line)
                output_file.write(line)

    # run the script from the correct working dir
    timer_start = datetime.datetime.now()
    result = subprocess.run(["python", temp_script_path], cwd=WORKING_DIR, capture_output=True)
    runtime = (datetime.datetime.now() - timer_start).total_seconds()
    total_runtime += runtime

    # insert info text into output image
    img = cv2.imread(SCRIPT_OUTPUT_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

    output_filename = f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_VAR_{variable_state}{Path(SCRIPT_OUTPUT_IMAGE_PATH).suffix}"
    cv2.imwrite(str(Path(OUTPUT_DIR, output_filename)), img)

    # print runtime for each script run
    logger.info(f"running variable: {variable_state:10} | total time: {runtime:>6.2f}s")

logger.info(f"total experiment runtime: {total_runtime:>6.2f}s")