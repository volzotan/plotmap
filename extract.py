from svgwriter import SvgWriter
from projection import Converter

import sys
import math
from datetime import datetime

# import xml.etree.ElementTree as ET  
import lxml.etree as ET

# export PYTHONIOENCODING=UTF-8

MAP_CENTER      = (50.980467, 11.325000)
MAP_SIZE        = 270 # units. may be px, may be mm

bounding_box = Converter.get_bounding_box_in_latlon(MAP_CENTER, 3000, 1400)

MAP_UP_LEFT = bounding_box[0]
MAP_DOWN_RIGHT = bounding_box[1]

conv = Converter(MAP_UP_LEFT, MAP_DOWN_RIGHT, MAP_SIZE)
svg = SvgWriter("test.svg", conv.get_map_size())

viewport_size = conv.get_map_size()
print("SVG dimensions: {0:.2f} x {1:.2f}mm".format(*viewport_size))

svg.add_hatching("default")
svg.add_hatching("dense", distance=1.0)

svg.add_layer("parks")
svg.add_layer("buildings")
svg.add_layer("streets")
svg.add_layer("water")

svg.add_layer("meta")
svg.add_layer("whatever")
svg.add_layer("marker")

tree = ET.parse("weimar.osm")  
# tree = ET.parse("weimar_theaterplatz.osm")  
root = tree.getroot()

# parse document

nodes = {}
for node in root.findall("./node"):
    nodes[node.attrib["id"]] = [float(node.attrib["lat"]), float(node.attrib["lon"])]

print("indexed {} nodes".format(len(nodes.keys())))

# --- MISC

svg.add_circles([conv.convert(*MAP_CENTER)], layer="meta")

xy1 = conv.convert(*MAP_UP_LEFT)
xy2 = conv.convert(*MAP_DOWN_RIGHT)

# svg.add_rectangle([xy1, [xy2[0] - xy1[0], xy2[1] - xy1[1]]], stroke_width=0.2, layer="meta")

# --- PROCESS COMBINED

timer_start = datetime.now()

parks = []

for way in root.findall("./way"):

    way_nodes = []
    for node_id in way.findall("./nd[@ref]"):
        way_nodes.append(conv.convert(*nodes[node_id.attrib["ref"]]))

    # if not conv.all_elements_inside_boundary(way_nodes):
    #     continue

    polygon = False
    if way_nodes[0] == way_nodes[-1]: # closed loop
        polygon = True

    tags = _parse_tags(way)

    keys = tags.keys()

    if "building" in keys:
        svg.add_polygon(way_nodes, layer="buildings", stroke_width=0.2, opacity=0, hatching="default")
        continue

    if "highway" in keys:
        if not polygon:
            svg.add_poly_line(way_nodes, stroke_width=0.2, layer="streets")
        continue

    if "leisure" in keys:
        if tags["leisure"] == "park":
            parks.append(way_nodes)
        continue

    if "landuse" in keys:
        if tags["landuse"] == "grass" or tags["landuse"] == "forest":
            parks.append(way_nodes)
        continue

    if "natural" in keys:
        if tags["natural"] == "grassland":
            parks.append(way_nodes)
        continue

    if "waterway" in keys:
        if polygon:
            svg.add_polygon(way_nodes, stroke_width=0.2, layer="water")
        else:
            svg.add_poly_line(way_nodes, stroke_width=0.2, layer="water")
        continue

    # print(tags)

for way_nodes in parks:
    svg.add_polygon(way_nodes, stroke_width=0.2, opacity=0, hatching="dense", layer="parks")

print("processing nodes finished in {0:.2f}s".format((datetime.now()-timer_start).total_seconds()))

# --- STREETS

# timer_start = datetime.now()

# # highways

# for way in root.findall("./way"):

#     if len(way.findall("./tag[@k='building']")) > 0:
#         continue

#     if len(way.findall("./tag[@k='building:part']")) > 0:
#         continue

#     if len(way.findall("./tag[@k='landuse']")) > 0:
#         continue

#     # print(ET.tostring(way).decode())

#     way_nodes = []

#     for node_id in way.findall("./nd[@ref]"):
#         way_nodes.append(conv.convert(*nodes[node_id.attrib["ref"]]))

#     if not conv.all_elements_inside_boundary(way_nodes):
#         continue

#     # if conv.all_elements_outside_boundary(way_nodes):
#     #     continue

#     if way_nodes[0] == way_nodes[-1]: # closed loop
#         # svg.add_polygon(way_nodes, layer="whatever", stroke_width=0.2, opacity=0.5)
#         pass
#     else:
#         svg.add_poly_line(way_nodes, stroke_width=0.2, layer="streets")

# print("processing streets finished in {0:.2f}s".format((datetime.now()-timer_start).total_seconds()))

# # --- BUILDINGS

# timer_start = datetime.now()

# added_polygons = 0
# filtered_polygons = 0
# for way in root.findall("./way/tag[@k='building']/.."):

#     polygon = []

#     for node_id in way.findall("./nd[@ref]"):
#         polygon.append(conv.convert(*nodes[node_id.attrib["ref"]]))

#     if not conv.all_elements_inside_boundary(polygon):
#         continue

#     # if area_of_polygon(polygon) < 0.01:
#     #     filtered_polygons += 1
#     #     continue

#     # if conv.all_elements_outside_boundary(polygon):
#     #     continue

#     added_polygons += 1
#     svg.add_polygon(polygon, layer="buildings", stroke_width=0.2, opacity=0, hatching="default")

# print("processing buildings finished in {0:.2f}s".format((datetime.now()-timer_start).total_seconds()))
# print("filtered buildings: {}/{}".format(filtered_polygons, filtered_polygons+added_polygons))

# --- MARKER

img_tag = "<image xlink:href=\"{}\" x=\"{}\" y=\"{}\" height=\"{}\" width=\"{}\"/>"

viewport_size = conv.get_map_size()

markersize = 30

marker_coordinates = {
    "1": ["marker_1.jpg", 0, 0],
    "2": ["marker_2.jpg", 0, viewport_size[1]-markersize],
    "3": ["marker_3.jpg", viewport_size[0]-markersize, 0],
    "4": ["marker_4.jpg", viewport_size[0]-markersize, viewport_size[1]-markersize]
}

for mid in marker_coordinates.keys():
    lat, lon = conv.convert_px_to_latlon(marker_coordinates[mid][1], marker_coordinates[mid][2])
    marker_coordinates[mid].append([lat, lon])

    svg.add_raw_element(img_tag.format(marker_coordinates[mid][0], marker_coordinates[mid][1], marker_coordinates[mid][2], markersize, markersize), layer="marker")

# print(marker_coordinates)

# --- 

svg.save()
