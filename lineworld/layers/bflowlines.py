import os
from dataclasses import dataclass
from pathlib import Path

import fiona
import geoalchemy2
import numpy as np
import rasterio
import shapely
from core.maptools import DocumentInfo, Projection
from geoalchemy2.shape import to_shape, from_shape
from layers.layer import Layer
from loguru import logger
from shapely import Polygon, MultiLineString, MultiPolygon, Point
from shapely.affinity import affine_transform
from shapely.geometry import shape
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds

from experiments.hatching import flowlines, scales
from experiments.hatching.flowlines import FlowlineTiler
from lineworld.util.geometrytools import hershey_text_to_lines, add_to_exclusion_zones

@dataclass
class BflowlinesMapLines():
    id: int | None
    lines: MultiLineString

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "lines": str(from_shape(self.lines))
        }

class Bflowlines(Layer):
    DATA_URL = ""
    DATA_SRID = Projection.WGS84

    LAYER_NAME = "BathymetryFlowlines"
    DATA_DIR = Path("data", LAYER_NAME.lower())

    SOURCE_FILE = Path("data", "elevation", "gebco_mosaic.tif") # TODO
    ELEVATION_FILE = Path(DATA_DIR, "flowlines_elevation.tif")

    # simplification tolerance in WGS84 latlon, resolution: 1°=111.32km (equator worst case)
    LAT_LON_PRECISION = 0.01
    LAT_LON_MIN_SEGMENT_LENGTH = 0.1

    EXCLUDE_BUFFER_DISTANCE = 2

    def __init__(self, layer_label: str, db: engine.Engine, px_per_mm: float = 10.0) -> None:
        super().__init__(layer_label, db)

        if not self.DATA_DIR.exists():
            os.makedirs(self.DATA_DIR)

        self.px_per_mm = px_per_mm
        self.raster_width = None
        self.raster_height = None

        metadata = MetaData()

        self.map_lines_table = Table("bflowlines_map_lines", metadata,
                                     Column("id", Integer, primary_key=True),
                                     Column("lines", geoalchemy2.Geometry("LINESTRING"), nullable=False)
                                     )

        metadata.create_all(self.db)

    def extract(self) -> None:
        pass

    def transform_to_world(self) -> None:
        pass

    def transform_to_map(self, document_info: DocumentInfo) -> None:

        logger.info("reprojecting GeoTiff")

        if self.ELEVATION_FILE.exists():
            logger.info("reprojected GeoTiff exists, skipping")
        else:
            with rasterio.open(self.SOURCE_FILE) as src:
                dst_crs = f"{document_info.projection.value[0]}:{document_info.projection.value[1]}"

                # calculate optimal output dimensions to get the width/height-ratio after reprojection
                _, dst_width, dst_height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
                ratio = dst_width / dst_height

                dst_width = int(document_info.width * self.px_per_mm)
                dst_height = int(document_info.width * self.px_per_mm * ratio)

                xmin, ymin, xmax, ymax = transform_bounds(src.crs, dst_crs, *src.bounds)
                dst_transform = rasterio.transform.Affine(
                    (xmax - xmin) / float(dst_width),
                    0, xmin, 0,
                    (ymin - ymax) / float(dst_height),
                    ymax
                )

                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height
                })

                with rasterio.open(self.ELEVATION_FILE, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        band_arr = src.read(i)

                        # remove any above-waterlevel terrain
                        band_arr[band_arr > 0] = 0

                        reproject(
                            source=band_arr,
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest
                        )

                logger.debug(f"reprojected fo file {self.ELEVATION_FILE} | {dst_width} x {dst_height}px")


        with rasterio.open(self.ELEVATION_FILE) as src:
            self.raster_width = src.width
            self.raster_height = src.height


    def transform_to_lines(self, document_info: DocumentInfo) -> list[BflowlinesMapLines]:

        config = flowlines.FlowlineHatcherConfig()

        data = None
        with rasterio.open(self.ELEVATION_FILE) as dataset:
            data = dataset.read(1)

        density_data = np.copy(data)
        density_normalized = (density_data - np.min(density_data)) / (np.max(density_data) - np.min(density_data))

        # Apply a non-linear scale
        scale = scales.Scale(scales.quadratic_bezier, {"p1": [0.30, 0], "p2": [.70, 1.0]})
        density_normalized = scale.apply(density_normalized)

        density = (np.full(density_data.shape, config.LINE_DISTANCE[0], dtype=float) +
                   (density_normalized * (config.LINE_DISTANCE[1] - config.LINE_DISTANCE[0])))

        tiler = FlowlineTiler(
            data,
            density,
            config,
            [4, 4]
        )

        linestrings = tiler.hatch()

        # convert from raster pixel coordinates to map coordinates
        mat = document_info.get_transformation_matrix_raster(self.raster_width, self.raster_height)
        linestrings = [affine_transform(line, mat) for line in linestrings]

        return [BflowlinesMapLines(None, line) for line in linestrings]

    def load(self, geometries: list[BflowlinesMapLines]) -> None:

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

    def out(self, exclusion_zones: MultiPolygon, document_info: DocumentInfo) -> tuple[
        list[shapely.Geometry], MultiPolygon]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

        # cut extrusion_zones into drawing_geometries

        drawing_geometries_cut = []
        stencil = shapely.difference(document_info.get_viewport(), exclusion_zones)
        for g in drawing_geometries:
            drawing_geometries_cut.append(shapely.intersection(g, stencil))

        # and do not add anything to exclusion_zones

        return (drawing_geometries_cut, exclusion_zones)
