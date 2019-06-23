from math import sqrt
import sys

# Speed in mm/min
TRAVEL_SPEED    = 5000 
WRITE_SPEED     = 5000
PEN_MOVE_SPEED  = 1 # in sec

pen_up          = 0
pen_down        = 0

state_travel    = False

distance_write  = 0
distance_travel = 0

filename = sys.argv[1]

with open(filename) as f: 
    for line in f: 

        if len(line) == 0:
            print("empty line")
        
        if line.startswith("G1 Z1") or line.startswith("G0 Z1"):
            pen_up += 1
            state_travel = True

        if line.startswith("G1 Z0") or line.startswith("G0 Z0"):
            pen_down += 1
            state_travel = False

        if line.startswith("G1  X") or line.startswith("G0  X"):
            s = line.split(" ")
            x = float(s[2][1:])
            y = float(s[3][1:])
            dist = sqrt(x**2 + y**2)

            # print("{}, {}".format(x, y))
            # print(dist)

            if state_travel:
                distance_travel += dist
            else:
                distance_write += dist

        # if line.startswith(""):
        # print (line) 

print("pen_up   events : {0:>10}".format(pen_up))
print("pen_down events : {0:>10}".format(pen_down))
print("distance travel : {0:>10.2f} m".format(distance_travel/10000.0))
print("distance  write : {0:>10.2f} m".format(distance_write/10000.0))

time_travel = distance_travel / TRAVEL_SPEED
time_write = distance_write / WRITE_SPEED
time_total = + time_travel + time_write + (PEN_MOVE_SPEED / 60.0)

print("time travel     : {0:>10.2f} min".format(time_travel))
print("time write      : {0:>10.2f} min".format(time_write))

print("time_total      : {0:>10.2f} min".format(time_total))