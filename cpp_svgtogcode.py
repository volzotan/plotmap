import lxml.etree as ET
import math
from datetime import datetime

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pandas as pd

# SVG_FILENAME = "test.svg"
# SVG_FILENAME = "dtm/weimar_50m.svg"
SVG_FILENAME = "dtm/thueringen_50m.svg"
# SVG_FILENAME = "shapely.svg"

MAX_LINES_PER_GRAPH = 100000

TRAVEL_SPEED = 5000
WRITE_SPEED = 2500

COMP_TOLERANCE = 0.001
MIN_LINE_LENGTH = 0.3 # in mm

OUTPUT_FILENAME = "test.gcode"

CMD_WRITE = "G1  X{0:.3f} Y{1:.3f}\n"
CMD_TRAVEL = "G0  X{0:.3f} Y{1:.3f}\n"
CMD_PEN_UP = "G0 Z1 F1000\n"
CMD_PEN_DOWN = "G0 Z0 F1000\n"

state_pen_up = True

OPTIMIZE_ORDER = True

tree = ET.parse(SVG_FILENAME)  
root = tree.getroot()

DEFAULT_NS = "{" + root.nsmap[None] + "}"
INKSCAPE_NS = "{" + root.nsmap["inkscape"] + "}"

TIMER_STRING = "{:<50s}: {:2.2f}s"

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


all_lines = []

for layer in root.findall("g", root.nsmap):
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

def add_augmenting_path_to_graph(graph, min_weight_pairs):
    """
    Add the min weight matching edges to the original graph
    Parameters:
        graph: NetworkX graph (original graph from trailmap)
        min_weight_pairs: list[tuples] of node pairs from min weight matching
    Returns:
        augmented NetworkX graph
    """
    
    # We need to make the augmented graph a MultiGraph so we can add parallel edges
    graph_aug = nx.MultiGraph(graph.copy())
    for pair in min_weight_pairs:
        graph_aug.add_edge(pair[0], 
                           pair[1], 
                           **{"type": "augmented"}
                           # **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]), 'type': 'augmented'}
                           # attr_dict={'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]),
                           #            'trail': 'augmented'}  # deprecated after 1.11
                          )
    return graph_aug

def create_complete_graph(pair_weights, flip_weights=True):
    """
    Create a completely connected graph using a list of vertex pairs and the shortest path distances between them
    Parameters: 
        pair_weights: list[tuple] from the output of get_shortest_paths_distances
        flip_weights: Boolean. Should we negate the edge attribute in pair_weights?
    """
    g = nx.Graph()
    for k, v in pair_weights.items():
        wt_i = - v if flip_weights else v
        # g.add_edge(k[0], k[1], {'distance': v, 'weight': wt_i})  # deprecated after NX 1.11 
        g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})  
    return g

def process_subgraph(subgraph):

    start_node = list(subgraph.nodes())[0]

    node_positions = {}
    for node in subgraph.nodes(data=True):
        node_positions[node[0]] = (node[1]["pos"][0], -node[1]["pos"][1])

    # print("------")
    # print("number of nodes: {}".format(len(subgraph.nodes())))

    nodes_odd_degree = [v for v, d in subgraph.degree() if d % 2 == 1]
    # print("# nodes odd degree: {}".format(len(nodes_odd_degree)))

    if len(nodes_odd_degree) > 0:

        # Compute all pairs of odd nodes. in a list of tuples
        odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))

        if (len(odd_node_pairs) > 1):
            print('# of odd pairs: {}'.format(len(odd_node_pairs)))

        odd_node_pairs_shortest_paths = {}
        for pair in odd_node_pairs:

            l1 = node_positions[pair[0]]
            l2 = node_positions[pair[1]]

            odd_node_pairs_shortest_paths[pair] = math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)

        # Generate the complete graph
        g_odd_complete = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)

        # Counts
        # print('Number of nodes: {}'.format(len(g_odd_complete.nodes())))
        # print('Number of edges: {}'.format(len(g_odd_complete.edges())))

        # Compute min weight matching.
        # Note: max_weight_matching uses the 'weight' attribute by default as the attribute to maximize.
        # odd_matching_dupes = list(nx.algorithms.max_weight_matching(g_odd_complete, True))
        odd_matching = list(nx.algorithms.max_weight_matching(g_odd_complete, True))
        # print('Number of edges in matching: {}'.format(len(odd_matching)))

        g_odd_complete_min_edges = nx.Graph(odd_matching)

        # Create augmented graph: add the min weight matching edges to g
        g_aug = add_augmenting_path_to_graph(subgraph, odd_matching)

        # Counts
        # print('Number of edges in original graph: {}'.format(len(subgraph.edges())))
        # print('Number of edges in augmented graph: {}'.format(len(g_aug.edges())))

        subgraph = g_aug

    naive_euler_circuit_edges = list(nx.eulerian_circuit(subgraph, source=start_node))

    lines = []

    for tup in naive_euler_circuit_edges:
        node_curr = subgraph.node[tup[0]]
        node_next = subgraph.node[tup[1]]

        # remove edge if it's the only edge and it's augmented
        # sometimes an edge may be present multiple times and one of those
        # times it's augmented, then let the edge be drawn n+1 times

        # TODO: improvement: I may want to have the line drawn exactly n, not n+1 times...

        data = subgraph.get_edge_data(tup[0], tup[1])
        if data is not None and len(data.keys()) == 1 and "type" in data[0]:
            continue

        from_pos = node_curr["pos"]
        to_pos = node_next["pos"]

        lines.append([from_pos[0], from_pos[1], to_pos[0], to_pos[1]])

    # print('Number of edges in euler path: {}'.format(len(lines)))

    return lines

timer_start = datetime.now()

NODE_NAME = "{:4.4f}|{:4.4f}"

number_lines = len(nplines)
ranges = []

if number_lines < MAX_LINES_PER_GRAPH:
    ranges = [[0, len(nplines)]]
else:
    rangenumber = int(number_lines/MAX_LINES_PER_GRAPH) + 1
    rangesize = int(number_lines/rangenumber)

    for i in range(0, rangenumber-1):
        ranges.append([rangesize*i, rangesize*(i+1)])

    ranges.append([rangesize * (rangenumber-1), len(nplines)])    

subgraphs = []

for r in ranges:

    g = nx.Graph()

    for i in range(r[0], r[1]):
        line = nplines[i]

        l1 = [line[0], line[1]]
        l2 = [line[2], line[3]]

        g.add_node(NODE_NAME.format(*l1), pos=l1)
        g.add_node(NODE_NAME.format(*l2), pos=l2)

        g.add_edge(NODE_NAME.format(*l1), NODE_NAME.format(*l2))

    print('# of edges: {}'.format(g.number_of_edges()))
    print('# of nodes: {}'.format(g.number_of_nodes()))

    subgraphs = subgraphs + list(nx.connected_component_subgraphs(g))

print(TIMER_STRING.format("adding edges and preparing graph", (datetime.now()-timer_start).total_seconds()))

lines_of_subgraphs = []
subgraphs_start_ends = []
for subgraph in subgraphs:

    timer_start = datetime.now()

    lines = process_subgraph(subgraph)

    print(TIMER_STRING.format("process subgraph", (datetime.now()-timer_start).total_seconds()))

    if len(lines) == 0:
        continue

    lines_of_subgraphs.append(lines)

    l1 = lines[0]
    l1 = [l1[0], l1[1]]

    l2 = lines[-1]
    l2 = [l2[2], l2[3]]

    subgraphs_start_ends.append([*l1, *l2])

def order_lines(lines):

    nplines = np.asarray(lines)

    ordered_indices = [0]
    indices_done_mask = np.zeros(nplines.shape[0], dtype=bool)
    indices_done_mask[ordered_indices] = True

    for i in range(1, nplines.shape[0]):

        last = lines[ordered_indices[-1]]
        indices_done_mask[ordered_indices] = True

        # manhattan distance

        distance_forw = np.add(np.abs(np.subtract(nplines[:, 0], last[2])), np.abs(np.subtract(nplines[:, 1], last[3])))
        distance_forw_masked = np.ma.masked_array(distance_forw, mask=indices_done_mask)
        distance_forw_min = np.argmin(distance_forw_masked)
        ordered_indices.append(distance_forw_min)

    return ordered_indices

ordered_lines = []
for index in order_lines(subgraphs_start_ends):
    ordered_lines = ordered_lines + lines_of_subgraphs[index]

# for i in range(1, len(lines_of_subgraphs)-1):

#     if len(lines_of_subgraphs[i-1]) == 0:
#         continue

#     l1 = lines_of_subgraphs[i-1][-1]
#     l1 = [l1[2], l1[3]]

#     min_index = None
#     min_distance = None

#     for j in range(i+1, len(lines_of_subgraphs)):

#         if len(lines_of_subgraphs[j]) == 0:
#             continue

#         l2 = lines_of_subgraphs[j][0]
#         l2 = [l2[2], l2[3]]

#         distance = math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)

#         if min_distance is None or distance < min_distance:
#             min_distance = distance
#             min_index = i+j

#     if min_index is not None:
#         tmp = lines_of_subgraphs[i]
#         lines_of_subgraphs[i] = lines_of_subgraphs[min_index]
#         lines_of_subgraphs[min_index] = tmp

# ordered_lines = []
# for lines in lines_of_subgraphs:
#     ordered_lines = ordered_lines + lines

# metagraph = nx.Graph()
# for lines in lines_of_subgraphs:

#     l1 = [lines[0][0], lines[0][1]]
#     l2 = [lines[-1][2], lines[-1][3]]

#     metagraph.add_node(NODE_NAME.format(*l1), pos=l1)
#     metagraph.add_node(NODE_NAME.format(*l2), pos=l2)

#     g.add_edge(NODE_NAME.format(*l1), NODE_NAME.format(*l2))

# print('metagraph # of edges: {}'.format(metagraph.number_of_edges()))
# print('metagraph # of nodes: {}'.format(metagraph.number_of_nodes()))

# node_positions = {}
# for node in g.nodes(data=True):
#     node_positions[node[0]] = (node[1]["pos"][0], -node[1]["pos"][1])

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

        if (state_pen_up):
            out.write(CMD_TRAVEL.format(line[0], line[1]))

            out.write("G1 Z0 F1000\n")
            out.write("G1 F{}\n".format(WRITE_SPEED))
            state_pen_up = False
        else:
            out.write(CMD_WRITE.format(line[0], line[1]))

        out.write(CMD_WRITE.format(line[2], line[3]))  

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
