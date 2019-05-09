# get data as a pbf (compressed binary protobuf format)
# and cut the region of interest (East-South-West-North bounding box)
# map for coordinates: https://informationfreeway.org/

# osmium extract -b 11.4048,50.9185,11.2166,51.0395 germany-latest.osm.pbf -o weimar.pbf

osmium extract -b 11.3271,50.9790,11.3248,50.9808 weimar.pbf -o weimar_theaterplatz.pbf

# afterwards convert to xml format
osmosis --read-pbf weimar_theaterplatz.pbf --write-xml weimar_theaterplatz.osm