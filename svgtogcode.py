import lxml.etree as ET
import math
from datetime import datetime

import numpy as np

tree = ET.parse("test.svg")  
root = tree.getroot()

DEFAULT_NS = "{" + root.nsmap[None] + "}"
INKSCAPE_NS = "{" + root.nsmap["inkscape"] + "}"

TRAVEL_SPEED = 3000
WRITE_SPEED = 2000

COMP_TOLERANCE = 0.001
MIN_LINE_LENGTH = 1.0 # in mm

OUTPUT_FILENAME = "test.gcode"

CMD_MOVE = "G1  X{0:.3f} Y{1:.3f}\n"
CMD_PEN_UP = "G1 Z1 F1000\n"

state_pen_up = True

# np.set_printoptions(precision=4,
#                        threshold=10000,
#                        linewidth=150)

np.set_printoptions(suppress=True)

def process(e):
    lines = []

    if e.tag == DEFAULT_NS + "rect":
        x1 = float(e.attrib["x"])
        y1 = float(e.attrib["y"])
        x2 = x1 + float(e.attrib["width"])
        y2 = y1 + float(e.attrib["height"])

        lines.append([x1, y1, x1, y2]) # left
        lines.append([x1, y1, x2, y1]) # top
        lines.append([x1, y2, x2, y2]) # bottom
        lines.append([x2, y1, x2, y2]) # right

        return lines

    if e.tag == DEFAULT_NS + "line":
        lines.append([float(e.attrib["x1"]), float(e.attrib["y1"]), float(e.attrib["x2"]), float(e.attrib["y2"])])
        return lines

    if e.tag == DEFAULT_NS + "path":
        d = e.attrib["d"]
        d = d[1:] # cut off the M
        segments = d.split("L")

        l = []

        for s in segments:
            pairs = s.split(" ")
            l.append([float(pairs[0]), float(pairs[1])])

        for i in range(1, len(l)):
            lines.append([l[i-1][0], l[i-1][1], l[i][0], l[i][1]])

        return lines

    if e.tag == DEFAULT_NS + "circle":
        print("invalid element: {}".format(e.tag))
        return lines

    if e.tag == DEFAULT_NS + "image":
        print("invalid element: {}".format(e.tag))
        return lines

    print("unknown element: {}".format(e.tag))
    return lines


def compare_equal(e0, e1):
    if math.isclose(e0[0], e1[0], abs_tol=COMP_TOLERANCE):
        if math.isclose(e0[1], e1[1], abs_tol=COMP_TOLERANCE):
            return True

    return False


all_lines = []

for layer in root.findall("g", root.nsmap):
    # print(layer.tag)
    # print(layer.attrib)

    for child in layer:
        all_lines = all_lines + process(child)


# all_lines = all_lines[0:3000]

print(" ")
print("--------------------------------------------------")
print(" ")

number_of_lines = len(all_lines)
print("number of lines: {}".format(number_of_lines))

# ------------------------------------------------------------------------------------
# filter duplicates

nplines = np.array(all_lines, dtype=np.float)
unique = np.unique(nplines, axis=0)

number_duplicates = len(all_lines) - unique.shape[0]
print("cleaned duplicates: {0} | duplicate ratio: {1:.2f}%".format(number_duplicates, (number_duplicates/len(all_lines))*100))

nplines = unique

# ------------------------------------------------------------------------------------
# filter tiny lines

# distances = np.sqrt(np.add(np.power(np.subtract(nplines[:, 0], nplines[:, 2]), 2), np.power(np.subtract(nplines[:, 1], nplines[:, 3]), 2)))
# indices_shortlines = np.where(distances < MIN_LINE_LENGTH)[0]
# nplines = np.delete(nplines, indices_shortlines, axis=0)
# print("cleaned short lines: {0} | short line ratio: {1:.2f}%".format(indices_shortlines.shape[0], (indices_shortlines.shape[0]/len(all_lines))*100))

# evil ... breaks paths without pen up/down events in two and creates gaps

# ------------------------------------------------------------------------------------
# mirror along x axis to transfer SVG coordinate system (0 top left) to gcode (0 bottom left)

maxy = np.max([np.max(nplines[:, 1]), np.max(nplines[:, 3])])

nplines[:, 1] = np.multiply(nplines[:, 1], -1)
nplines[:, 3] = np.multiply(nplines[:, 3], -1)

nplines[:, 1] = np.add(nplines[:, 1], maxy)
nplines[:, 3] = np.add(nplines[:, 3], maxy)

# ------------------------------------------------------------------------------------
# optimize drawing order. greedy (and inefficient)

timer = datetime.now()

indices_done = [0]
# indices_done_mask = np.zeros(nplines.shape, dtype=bool)
indices_done_mask = np.zeros(nplines.shape[0], dtype=bool)
# indices_done_mask[indices_done, :] = True
indices_done_mask[indices_done] = True
ordered_lines = [nplines[0, :]]

# nplines_masked = np.ma.masked_array(nplines, mask=indices_done_mask)

for i in range(0, nplines.shape[0]):

    print("{0:.2f}".format((len(ordered_lines)/nplines.shape[0])*100.0), end="\r")

    last = ordered_lines[-1]
    # indices_done_mask[indices_done, :] = True
    indices_done_mask[indices_done] = True

    distance_forw = np.sqrt(np.add(np.power(np.subtract(nplines[:, 0], last[2]), 2), np.power(np.subtract(nplines[:, 1], last[3]), 2)))
    distance_back = np.sqrt(np.add(np.power(np.subtract(nplines[:, 2], last[2]), 2), np.power(np.subtract(nplines[:, 3], last[3]), 2)))

    distance_forw_masked = np.ma.masked_array(distance_forw, mask=indices_done_mask)
    distance_back_masked = np.ma.masked_array(distance_back, mask=indices_done_mask)

    distance_forw_min = np.argmin(distance_forw_masked)
    distance_back_min = np.argmin(distance_back_masked)

    if distance_forw[distance_forw_min] < distance_back[distance_back_min]:
        indices_done.append(distance_forw_min)
        ordered_lines.append(nplines[distance_forw_min, :])
    else:
        indices_done.append(distance_back_min)
        flip = nplines[distance_back_min, :]
        ordered_lines.append(np.array([flip[2], flip[3], flip[0], flip[1]]))

print("optimization done. time: {0:.2f}s".format((datetime.now()-timer).total_seconds()))

# ------------------------------------------------------------------------------------
# filter tiny edges/leaves/whatever (small lines which are not connected)

nplines = np.array(ordered_lines, dtype=np.float)

distances = np.sqrt(np.add(np.power(np.subtract(nplines[:, 0], nplines[:, 2]), 2), np.power(np.subtract(nplines[:, 1], nplines[:, 3]), 2)))
indices_shortlines = np.where(distances < MIN_LINE_LENGTH)[0]

unconnected_indices = []
for i in range(1, nplines.shape[0]-1):
    prv = nplines[i-1, :] 
    cur = nplines[i  , :]
    nxt = nplines[i+1, :]

    if not prv[2] == cur[0] or not prv[3] == cur[1] or not cur[2] == nxt[0] or not cur[3] == nxt[1]:
        if i in indices_shortlines:
            unconnected_indices.append(i)

nplines = np.delete(nplines, unconnected_indices, axis=0)
ordered_lines = nplines

print("cleaned unconnected short lines: {0} | short line ratio: {1:.2f}%".format(len(unconnected_indices), (len(unconnected_indices)/len(all_lines))*100))

# ------------------------------------------------------------------------------------


# ordered_lines = []
# ordered_lines.append(all_lines[0])
# all_lines.remove(ordered_lines[0])

# while(len(all_lines) > 0):

#     print("{0:.2f}".format((len(ordered_lines)/number_of_lines)*100.0))

#     src = ordered_lines[-1][1]
#     dst = all_lines[0]

#     candidate = dst
#     candidate_distance = distance(src, candidate[0])
#     candidateFlip = False

#     candidate_i = 0

#     for i in range(0, len(all_lines)):
#         dst = all_lines[i]

#         distance0 = distance(src, dst[0])
#         distance1 = distance(src, dst[1])

#         if distance0 < 0.001:
#             candidate = dst
#             candidateFlip = False
#             candidate_i = i
#             break

#         if distance1 < 0.001:
#             candidate = dst
#             candidateFlip = True
#             candidate_i = i
#             break

#         if distance0 < candidate_distance:
#             candidate = dst
#             candidate_distance = distance0
#             candidateFlip = False

#             candidate_i = i

#         if distance1 < candidate_distance:
#             candidate = dst
#             candidate_distance = distance1
#             candidateFlip = True

#             candidate_i = i

#     # print("{}|{} {}".format(candidate_i, len(all_lines), candidateFlip))

#     if candidateFlip:
#         ordered_lines.append([candidate[1], candidate[0]])
#     else:
#         ordered_lines.append(candidate)

#     all_lines.pop(candidate_i)

# print("number of ordered_lines: {}".format(len(ordered_lines)))

# print(order_index)

with open(OUTPUT_FILENAME, "w") as out:
    out.write("G90\n")                          # absolute positioning
    out.write("G21\n")                          # Set Units to Millimeters
    out.write(CMD_PEN_UP)                       # move pen up
    out.write("G1 F{}\n".format(TRAVEL_SPEED))  # Set feedrate to TRAVEL_SPEED mm/min
    state_pen_up = True
    out.write("\n")

    number_lines = len(ordered_lines)

    for i in range(0, number_lines):
        line = ordered_lines[i]
        line_next = None
        if (i + 1) < number_lines:
            line_next = ordered_lines[i+1]

        out.write(CMD_MOVE.format(line[0], line[1]))

        # pen down
        if (state_pen_up):
            out.write("G1 Z0 F1000\n")
            out.write("G1 F{}\n".format(WRITE_SPEED))
            state_pen_up = False

        out.write(CMD_MOVE.format(line[2], line[3]))  

        move_pen_up = True
        if line_next is not None:
            if math.isclose(line[2], line_next[0], abs_tol=COMP_TOLERANCE):
                if math.isclose(line[3], line_next[1], abs_tol=COMP_TOLERANCE):
                    move_pen_up = False
        if move_pen_up:
            out.write(CMD_PEN_UP)  
            out.write("G1 F{}\n".format(TRAVEL_SPEED))
            out.write("\n")
            state_pen_up = True

    out.write(CMD_PEN_UP)
    out.write("G1 F{}\n".format(TRAVEL_SPEED))
    out.write("G1  X{} Y{}".format(0, 0))
