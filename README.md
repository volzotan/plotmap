# Plotmap: a line-based world map generator

Create vector line (world) maps suitable for pen plotters. Data sources include [GEBCO](https://www.gebco.net/) elevation data 
and [OpenStreetMap coastlines](https://osmdata.openstreetmap.de/data/coastlines.html).

![header](media/header.png)

### Prerequisites

* Postgres including postGIS
* Python >= 3.12
* ~30 GB of disk space

### Run

Build a map with a rather small config file:   
```python3 lineworld/run.py```  
This may take a while to download the GEBCO elevation data. 

Convert the output SVG to gcode files, layer by layer:  
```python3 svgtogcode.py Lineworld750x500.svg```

### Learn more:

You can find more info in a blog post about how this came to be: [A Giant Map drawn with a Pen](https://volzo.de/posts/plottermap/)