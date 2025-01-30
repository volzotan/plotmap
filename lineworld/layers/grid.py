import itertools
from dataclasses import dataclass
from typing import Any

import geoalchemy2
import numpy as np
import shapely
from HersheyFonts import HersheyFonts
from core.maptools import DocumentInfo, Projection
from geoalchemy2.shape import from_shape, to_shape
from layers.layer import Layer
from loguru import logger
from shapely import envelope, MultiLineString, MultiPolygon, LineString, Point, Polygon
from shapely.affinity import affine_transform
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text

from lineworld.util import geometrytools
from lineworld.util.geometrytools import hershey_text_to_lines, add_to_exclusion_zones


@dataclass
class GridMapLines():
    id: int | None
    lines: MultiLineString

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "lines": str(from_shape(self.lines))
        }


class Grid(Layer):
    DATA_SRID = Projection.WGS84

    LAT_LON_MIN_SEGMENT_LENGTH = 1e-1

    DEFAULT_LATITUDE_LINE_DIST = 20
    DEFAULT_LONGITUDE_LINE_DIST = 20

    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

    def extract(self) -> None:
        pass

    def transform(self) -> None:
        pass

    def load(self, geometries: list[GridMapLines]) -> None:

        if geometries is None:
            return

        if len(geometries) == 0:
            logger.warning("no geometries to load. abort")
            return
        else:
            logger.info(f"loading geometries: {len(geometries)}")

        with self.db.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {self.map_lines_table.fullname} CASCADE"))
            conn.execute(insert(self.map_lines_table), [g.todict() for g in geometries])

    def _get_gridminmax(self, document_info: DocumentInfo) -> tuple[list[float], list[float]]:

        minmax_lat = [-90, 90]
        minmax_lon = [-180, 180]

        if document_info.wrapover:
            minmax_lon = [-180 - 90, 180 + 90]

        return minmax_lat, minmax_lon

    def _get_gridpositions(self, document_info: DocumentInfo, distance_lat_lines: float, distance_lon_lines: float) -> \
            tuple[list[float], list[float]]:

        minmax_lat, minmax_lon = self._get_gridminmax(document_info)

        lons = [x * distance_lat_lines for x in range(1, (minmax_lon[1] // distance_lat_lines)+1)]
        lons = list(reversed([x * -1 for x in lons])) + [0] + lons

        if not lons[0] == minmax_lon[0]:
            lons = [minmax_lon[0]] + lons + [minmax_lon[1]]

        lats = [x * distance_lon_lines for x in range(1, (minmax_lat[1] // distance_lon_lines)+1)]
        lats = list(reversed(lats)) + [0] + [x * -1 for x in lats]

        if not lats[0] == minmax_lat[1]:
            lats = [minmax_lat[1]] + lats + [minmax_lat[0]]

        return lats, lons

    def _get_gridlines(self, document_info: DocumentInfo, distance_lat_lines: float, distance_lon_lines: float) -> list[
        GridMapLines]:

        lines = []

        project_func = document_info.get_projection_func(self.DATA_SRID)
        mat = document_info.get_transformation_matrix()

        minmax_lat, minmax_lon = self._get_gridminmax(document_info)
        lats, lons = self._get_gridpositions(document_info, distance_lat_lines, distance_lon_lines)

        for lat in lats:
            lines.append(LineString([
                [minmax_lon[0], lat],
                [minmax_lon[1], lat]
            ]))

        for lon in lons:
            lines.append(LineString([
                [lon, minmax_lat[0]],
                [lon, minmax_lat[1]]
            ]))

        lines = shapely.segmentize(lines, self.LAT_LON_MIN_SEGMENT_LENGTH)

        lines = [shapely.ops.transform(project_func, line) for line in lines]
        lines = [affine_transform(line, mat) for line in lines]

        return [GridMapLines(None, line) for line in lines]

    def _get_grid_polygons(self, document_info: DocumentInfo, distance_lat_lines: float, distance_lon_lines: float) -> list[
        Polygon]:

        project_func = document_info.get_projection_func(self.DATA_SRID)
        mat = document_info.get_transformation_matrix()

        minmax_lat, minmax_lon = self._get_gridminmax(document_info)
        lats, lons = self._get_gridpositions(document_info, distance_lat_lines, distance_lon_lines)

        polys = []
        for index_lat in range(len(lats)-1):
            for index_lon in range(len(lons)-1):
                polys.append(shapely.box(
                    lons[index_lon],
                    lats[index_lat],
                    lons[index_lon+1],
                    lats[index_lat+1]
                ))

        polys = shapely.segmentize(polys, self.LAT_LON_MIN_SEGMENT_LENGTH)
        polys = [shapely.ops.transform(project_func, poly) for poly in polys]
        polys = [affine_transform(poly, mat) for poly in polys]

        viewport = shapely.box(0, 0, document_info.width, document_info.height) # TODO: use document_info.get_viewport()

        polys_cropped = []
        for poly in polys:
            cropped = shapely.intersection(viewport, poly)

            if type(cropped) is MultiPolygon:
                g = geometrytools.unpack_multipolygon(cropped)
                polys_cropped += g
            else:
                polys_cropped.append(cropped)

        polys_cropped = list(itertools.filterfalse(shapely.is_empty, polys_cropped))

        return polys_cropped

        # return polys

    def transform_to_map(self, document_info: DocumentInfo) -> list[GridMapLines]:
        pass

    def transform_to_lines(self, document_info: DocumentInfo) -> None:
        pass

    def _style(self, polygons: np.ndarray, document_info: DocumentInfo) -> None:
        pass

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        raise NotImplementedError("Must override method")


class GridBathymetry(Grid):
    LAYER_NAME = "GridBathymetry"

    DEFAULT_EXCLUDE_BUFFER_DISTANCE = 0.6

    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)
        
        metadata = MetaData()

        self.map_lines_table = Table(
            "gridbathymetry_map_lines", metadata,
            Column("id", Integer, primary_key=True),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
        )

        metadata.create_all(self.db)

    def transform_to_lines(self, document_info: DocumentInfo) -> list[GridMapLines]:
        return self._get_gridlines(
            document_info,
            self.config.get("latitude_line_dist", self.DEFAULT_LATITUDE_LINE_DIST),
            self.config.get("longitude_line_dist", self.DEFAULT_LONGITUDE_LINE_DIST)
        )

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

        # extend extrusion zones
        exclusion_zones = add_to_exclusion_zones(
            drawing_geometries,
            exclusion_zones,
            self.config.get("exclude_buffer_distance", self.DEFAULT_EXCLUDE_BUFFER_DISTANCE),
            self.config.get("tolerance", 0.1)
        )

        return ([], exclusion_zones)

    def get_polygons(self, document_info: DocumentInfo) -> list[Polygon]:
        return self._get_grid_polygons(
            document_info,
            self.config.get("latitude_line_dist", self.DEFAULT_LATITUDE_LINE_DIST),
            self.config.get("longitude_line_dist", self.DEFAULT_LONGITUDE_LINE_DIST)
        )


class GridLabels(Grid):
    LAYER_NAME = "GridLabels"

    DEFAULT_EXCLUDE_BUFFER_DISTANCE = 2.0

    DEFAULT_FONT_SIZE = 5

    OFFSET_TOP = 10
    OFFSET_BOTTOM = OFFSET_TOP
    OFFSET_LEFT = 5
    OFFSET_RIGHT = OFFSET_LEFT

    def __init__(self, layer_id: str, db: engine.Engine, config: dict[str, Any]) -> None:
        super().__init__(layer_id, db, config)

        metadata = MetaData()

        self.map_lines_table = Table(
            "gridlabels_map_lines", metadata,
            Column("id", Integer, primary_key=True),
            Column("lines", geoalchemy2.Geometry("MULTILINESTRING"), nullable=False)
        )

        metadata.create_all(self.db)

        self.hfont = HersheyFonts()
        self.hfont.load_default_font("futural")
        self.hfont.normalize_rendering(self.config.get("font_size", self.DEFAULT_FONT_SIZE))

    def transform_to_lines(self, document_info: DocumentInfo) -> list[GridMapLines]:

        gridlines = self._get_gridlines(
            document_info,
            self.config.get("latitude_line_dist", self.DEFAULT_LATITUDE_LINE_DIST),
            self.config.get("longitude_line_dist", self.DEFAULT_LONGITUDE_LINE_DIST))
        lats, lons = self._get_gridpositions(
            document_info,
            self.config.get("latitude_line_dist", self.DEFAULT_LATITUDE_LINE_DIST),
            self.config.get("longitude_line_dist", self.DEFAULT_LONGITUDE_LINE_DIST))

        project_func = document_info.get_projection_func(self.DATA_SRID)
        mat = document_info.get_transformation_matrix()

        labels = []

        for lon in lons:

            line_label = lon
            if line_label < -180:
                line_label = line_label + 360
            if line_label > +180:
                line_label = line_label - 360

            lines = hershey_text_to_lines(self.hfont, f"{line_label}")

            lon_line = LineString([[lon, -90], [lon, +90]])
            lon_line = shapely.segmentize(lon_line, self.LAT_LON_MIN_SEGMENT_LENGTH)

            lon_line = shapely.ops.transform(project_func, lon_line)
            lon_line = affine_transform(lon_line, mat)

            # TOP

            intersect_point_top = lon_line.intersection(
                LineString([[0, self.OFFSET_TOP], [document_info.width, self.OFFSET_TOP]])
            )

            if intersect_point_top is None or intersect_point_top.is_empty:
                logger.warning("failed computing position for label")
                continue

            center_offset = -envelope(lines).centroid.x

            mat_font = document_info.get_transformation_matrix_font(
                xoff=intersect_point_top.x + center_offset,
                yoff=intersect_point_top.y
            )

            labels.append(GridMapLines(None, affine_transform(lines, mat_font)))

            # BOTTOM

            intersect_point_bottom = lon_line.intersection(
                LineString([
                    [0, document_info.height - self.OFFSET_BOTTOM + self.config.get("font_size", self.DEFAULT_FONT_SIZE)],
                    [document_info.width, document_info.height - self.OFFSET_BOTTOM + self.config.get("font_size", self.DEFAULT_FONT_SIZE)]
                ])
            )

            if intersect_point_bottom is None or intersect_point_bottom.is_empty:
                logger.warning("failed computing position for label")
                continue

            mat_font = document_info.get_transformation_matrix_font(
                xoff=intersect_point_bottom.x + center_offset,
                yoff=intersect_point_bottom.y
            )

            labels.append(GridMapLines(None, affine_transform(lines, mat_font)))

        for lat in lats:

            lines = hershey_text_to_lines(self.hfont, f"{lat}")

            min_lon = -180
            max_lon = 180

            if document_info.wrapover:
                min_lon = -180 - 90
                max_lon = +180 + 90

            lat_line = LineString([[min_lon, lat], [max_lon, lat]])
            lat_line = shapely.segmentize(lat_line, self.LAT_LON_MIN_SEGMENT_LENGTH)

            lat_line = shapely.ops.transform(project_func, lat_line)
            lat_line = affine_transform(lat_line, mat)

            # LEFT

            intersect_point_left = lat_line.intersection(
                LineString([[self.OFFSET_LEFT, 0], [self.OFFSET_LEFT, document_info.height]])
            )

            if type(intersect_point_left) is not Point or intersect_point_left is None or intersect_point_left.is_empty:
                logger.warning(f"failed computing position for label [lat: {lat}]")
                continue

            center_offset = +envelope(lines).centroid.y

            mat_font = document_info.get_transformation_matrix_font(
                xoff=intersect_point_left.x,
                yoff=intersect_point_left.y + center_offset
            )

            labels.append(GridMapLines(None, affine_transform(lines, mat_font)))

            # RIGHT

            intersect_point_right = lat_line.intersection(
                LineString([
                    [document_info.width - self.OFFSET_RIGHT, 0],
                    [document_info.width - self.OFFSET_RIGHT, document_info.height]
                ])
            )

            if type(intersect_point_right) is not Point or intersect_point_right is None or intersect_point_right.is_empty:
                logger.warning(f"failed computing position for label [lat: {lat}]")
                continue

            center_offset = envelope(lines).centroid.xy

            mat_font = document_info.get_transformation_matrix_font(
                xoff=intersect_point_right.x - center_offset[0][0] * 2,
                yoff=intersect_point_right.y + center_offset[1][0]
            )

            labels.append(GridMapLines(None, affine_transform(lines, mat_font)))

        # return gridlines + labels
        return labels

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

        # remove extrusion zones
        drawing_geometries_cut = []
        for g in drawing_geometries:
            drawing_geometries_cut.append(shapely.difference(g, exclusion_zones))

        # extend extrusion zones
        exclusion_zones = add_to_exclusion_zones(
            drawing_geometries, exclusion_zones,
            self.config.get("exclude_buffer_distance", self.DEFAULT_EXCLUDE_BUFFER_DISTANCE),
            self.config.get("tolerance", 0.1))

        return (drawing_geometries_cut, exclusion_zones)
