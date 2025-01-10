import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
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

import lineworld
from experiments.hatching import flowlines, scales
from experiments.hatching.flowlines import FlowlineTiler
from lineworld.util.geometrytools import hershey_text_to_lines, add_to_exclusion_zones

@dataclass
class BathymetryFlowlinesMapLines():
    id: int | None
    lines: MultiLineString

    def todict(self) -> dict[str, int | float | str | None]:
        return {
            "lines": str(from_shape(self.lines))
        }

class BathymetryFlowlines(Layer):
    DATA_URL = ""
    DATA_SRID = Projection.WGS84

    LAYER_NAME = "BathymetryFlowlines"
    DATA_DIR = Path("data", LAYER_NAME.lower())

    SOURCE_FILE = Path("data", "elevation", "gebco_mosaic.tif") # TODO
    ELEVATION_FILE = Path(DATA_DIR, "flowlines_elevation.tif")
    DENSITY_FILE = Path("blender", "output.png")

    def __init__(self, layer_label: str, db: engine.Engine, config: dict[str, Any]={}) -> None:
        super().__init__(layer_label, db)

        if not self.DATA_DIR.exists():
            os.makedirs(self.DATA_DIR)

        self.config = config.get("layer", {}).get("bathymetryflowlines", {})

        metadata = MetaData()

        self.map_lines_table = Table("bathymetryflowlines_map_lines", metadata,
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

        with rasterio.open(self.SOURCE_FILE) as src:
            dst_crs = f"{document_info.projection.value[0]}:{document_info.projection.value[1]}"

            # calculate optimal output dimensions to get the width/height-ratio after reprojection
            _, dst_width, dst_height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            ratio = dst_width / dst_height

            px_per_mm = self.config.get("px_per_mm", 10.0)
            dst_width = int(document_info.width * px_per_mm)
            dst_height = int(document_info.width * px_per_mm * ratio)

            skip = False
            if self.ELEVATION_FILE.exists():
                with rasterio.open(self.ELEVATION_FILE) as dst:
                    if dst.width == dst_width and dst.height == dst_height:
                        skip = True
                        logger.info("reprojected GeoTiff exists, skipping")

            if not skip:

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


    def transform_to_lines(self, document_info: DocumentInfo) -> list[BathymetryFlowlinesMapLines]:

        flow_config = flowlines.FlowlineHatcherConfig()
        flow_config = lineworld.apply_config_to_object(self.config, flow_config)

        data = None
        with rasterio.open(self.ELEVATION_FILE) as dataset:
            data = dataset.read(1)

        data = cv2.resize(data, [15000, 15000]) # TODO

        density = None
        try:
            density = cv2.imread(str(self.DENSITY_FILE), cv2.IMREAD_GRAYSCALE)
            density = cv2.normalize(density, density, 0, 255, cv2.NORM_MINMAX).astype(np.float64)/255.0
            density = cv2.resize(density, data.shape)
        except Exception as e:
            logger.error(e)

        flow_config.MM_TO_PX_CONVERSION_FACTOR = data.shape[1] / document_info.width
        # flow_config.MM_TO_PX_CONVERSION_FACTOR = 5

        tiler = FlowlineTiler(data, density, flow_config,(self.config.get("num_tiles", 4), self.config.get("num_tiles", 4)))
        linestrings = tiler.hatch()

        # convert from raster pixel coordinates to map coordinates
        mat = document_info.get_transformation_matrix_raster(data.shape[1], data.shape[0])
        linestrings = [affine_transform(line, mat) for line in linestrings]
        linestrings = [line.simplify(self.config.get("tolerance", 0.1)) for line in linestrings]

        return [BathymetryFlowlinesMapLines(None, line) for line in linestrings]

    def load(self, geometries: list[BathymetryFlowlinesMapLines]) -> None:

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
