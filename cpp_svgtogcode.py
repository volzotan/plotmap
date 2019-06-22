import lxml.etree as ET
import math
from datetime import datetime

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pandas as pd

# SVG_FILENAME = "test.svg"
SVG_FILENAME = "dtm/elevation_lines.svg"

TRAVEL_SPEED = 5000
WRITE_SPEED = 2500

COMP_TOLERANCE = 0.001
MIN_LINE_LENGTH = 0.3 # in mm

OUTPUT_FILENAME = "test.gcode"

CMD_MOVE = "G1  X{0:.3f} Y{1:.3f}\n"
CMD_PEN_UP = "G1 Z1 F1000\n"

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

        # add last closing line
        lines.append([lines[-1][2], lines[-1][3], l[0][0], l[0][1]])

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

ordered_lines = None

timer_start = datetime.now()

NODE_NAME = "{:4.4f}|{:4.4f}"

g = nx.Graph()
for i in range(0, len(nplines)):
    line = nplines[i]

    l1 = [line[0], line[1]]
    l2 = [line[2], line[3]]

    distance = math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)

    g.add_node(NODE_NAME.format(*l1), pos=l1)
    g.add_node(NODE_NAME.format(*l2), pos=l2)

    g.add_edge(NODE_NAME.format(*l1), NODE_NAME.format(*l2), weight=distance)

print(TIMER_STRING.format("adding edges", (datetime.now()-timer_start).total_seconds()))

start_node = list(g.nodes())[0]

print('# of edges: {}'.format(g.number_of_edges()))
print('# of nodes: {}'.format(g.number_of_nodes()))

node_positions = {}
for node in g.nodes(data=True):
    node_positions[node[0]] = (node[1]["pos"][0], -node[1]["pos"][1])


complete_graph = g
subgraphs = list(nx.connected_component_subgraphs(g))

g = subgraphs[0]

# plt.figure(figsize=(8, 6))
# nx.draw(g, pos=node_positions, node_size=10, node_color='black')
# plt.show()

nodes_odd_degree = [v for v, d in g.degree() if d % 2 == 1]

print("# nodes odd degree: {}".format(len(nodes_odd_degree)))

# Compute all pairs of odd nodes. in a list of tuples
odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))

# Preview pairs of odd degree nodes
# print(odd_node_pairs[0:10])

print('# of odd pairs: {}'.format(len(odd_node_pairs)))

odd_node_pairs_shortest_paths = {}

for pair in odd_node_pairs:

    l1 = node_positions[pair[0]]
    l2 = node_positions[pair[1]]

    odd_node_pairs_shortest_paths[pair] = math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)

# print(dict(list(odd_node_pairs_shortest_paths.items())[0:10]))

# nx.draw(g, node_positions, node_size=2, node_color='k')
# # path_edges = zip(shortest_path, shortest_path[1:])
# # path_edges = set(path_edges)
# nx.draw_networkx_nodes(g, node_positions, nodelist=shortest_path, node_size=1, node_color='r')
# # nx.draw_networkx_edges(g, node_positions, edgelist=path_edges, edge_color='r', width=10)
# plt.show()

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

# Generate the complete graph
g_odd_complete = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)

# Counts
print('Number of nodes: {}'.format(len(g_odd_complete.nodes())))
print('Number of edges: {}'.format(len(g_odd_complete.edges())))

# Plot the complete graph of odd-degree nodes
# plt.figure(figsize=(8, 6))
# pos_random = nx.random_layout(g_odd_complete)
# nx.draw_networkx_nodes(g_odd_complete, node_positions, node_size=20, node_color="red")
# nx.draw_networkx_edges(g_odd_complete, node_positions, alpha=0.1)
# plt.axis('off')
# plt.title('Complete Graph of Odd-degree Nodes')
# plt.show()

# Compute min weight matching.
# Note: max_weight_matching uses the 'weight' attribute by default as the attribute to maximize.
# odd_matching_dupes = list(nx.algorithms.max_weight_matching(g_odd_complete, True))
odd_matching = list(nx.algorithms.max_weight_matching(g_odd_complete, True))
print('Number of edges in matching: {}'.format(len(odd_matching)))

# plt.figure(figsize=(8, 6))
# nx.draw(g_odd_complete, pos=node_positions, node_size=20, alpha=0.05)
g_odd_complete_min_edges = nx.Graph(odd_matching)
# nx.draw(g_odd_complete_min_edges, pos=node_positions, node_size=20, edge_color='blue', node_color='red')
# plt.title('Min Weight Matching on Complete Graph')
# plt.show()

# plt.figure(figsize=(8, 6))
# nx.draw(g, pos=node_positions, node_size=20, alpha=0.1, node_color='black')
# nx.draw(g_odd_complete_min_edges, pos=node_positions, node_size=20, alpha=1, node_color='red', edge_color='blue')
# plt.title('Min Weight Matching on Orginal Graph')
# plt.show()

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
                           **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]), 'trail': 'augmented'}
                           # attr_dict={'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]),
                           #            'trail': 'augmented'}  # deprecated after 1.11
                          )
    return graph_aug

# Create augmented graph: add the min weight matching edges to g
g_aug = add_augmenting_path_to_graph(g, odd_matching)

# Counts
print('Number of edges in original graph: {}'.format(len(g.edges())))
print('Number of edges in augmented graph: {}'.format(len(g_aug.edges())))

naive_euler_circuit = list(nx.eulerian_circuit(g_aug, source=start_node))

# print(naive_euler_circuit)

ordered_lines = []

for tup in naive_euler_circuit:
    node_curr = g_aug.node[tup[0]]
    node_next = g_aug.node[tup[1]]
    if "trail" in node_curr.keys():
        continue

    from_pos = node_curr["pos"]
    to_pos = node_next["pos"]

    ordered_lines.append([from_pos[0], from_pos[1], to_pos[0], to_pos[1]])

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
