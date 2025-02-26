"""

    The main 'geoname' table has the following fields :
    ---------------------------------------------------
 0  geonameid         : integer id of record in geonames database
 1  name              : name of geographical point (utf8) varchar(200)
 2  asciiname         : name of geographical point in plain ascii characters, varchar(200)
 3  alternatenames    : alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000)
 4  latitude          : latitude in decimal degrees (wgs84)
 5  longitude         : longitude in decimal degrees (wgs84)
 6  feature class     : see http://www.geonames.org/export/codes.html, char(1)
 7  feature code      : see http://www.geonames.org/export/codes.html, varchar(10)
 8  country code      : ISO-3166 2-letter country code, 2 characters
 9  cc2               : alternate country codes, comma separated, ISO-3166 2-letter country code, 200 characters
10  admin1 code       : fipscode (subject to change to iso code), see exceptions below, see file admin1Codes.txt for display names of this code; varchar(20)
11  admin2 code       : code for the second administrative division, a county in the US, see file admin2Codes.txt; varchar(80)
12  admin3 code       : code for third level administrative division, varchar(20)
13  admin4 code       : code for fourth level administrative division, varchar(20)
14  population        : bigint (8 byte int)
15  elevation         : in meters, integer
16  dem               : digital elevation model, srtm3 or gtopo30, average elevation of 3''x3'' (ca 90mx90m) or 30''x30'' (ca 900mx900m) area in meters, integer. srtm processed by cgiar/ciat.
17  timezone          : the iana timezone id (see file timeZone.txt) varchar(40)
18  modification date : date of last modification in yyyy-MM-dd format

"""

import csv
from pathlib import Path

INPUT_FILE = "cities15000.txt"
INPUT_FILE_ALTNAMES = "alternateNames.txt"
OUTPUT_FILE = "cities.csv"

MIN_POPULATION = 100_000
ALTNAME_COUNTRYCODE = "de"

altnames_map = {}

with open(Path(INPUT_FILE_ALTNAMES)) as file:
    for line in file:
        fields = line.split("\t")

        if not fields[2].lower() == ALTNAME_COUNTRYCODE:
            continue

        altnames_map[fields[1]] = fields[3]


with open(Path(INPUT_FILE)) as file_read:
    with open(Path(OUTPUT_FILE), "w") as file_write:
        writer = csv.DictWriter(
            file_write,
            delimiter=";",
            fieldnames=["ascii_name", "altname", "population", "lon", "lat"],
        )
        writer.writeheader()
        for line in file_read:
            fields = line.split("\t")

            if int(fields[14]) < MIN_POPULATION:
                continue

            row = {
                "ascii_name": fields[2],
                "altname": altnames_map.get(fields[0], ""),
                "population": fields[14],
                "lon": fields[5],
                "lat": fields[4],
            }

            writer.writerow(row)
