import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import geoalchemy2
import numpy as np
import rasterio
import shapely
from lineworld.core.map import DocumentInfo, Projection
from geoalchemy2.shape import to_shape, from_shape
from lineworld.layers.layer import Layer
from loguru import logger
from scipy import ndimage
from shapely import Polygon, MultiLineString, STRtree, LineString, GeometryCollection
from shapely.affinity import affine_transform
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer
from sqlalchemy import engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import text
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)

import lineworld
from lineworld.core import flowlines
from lineworld.core.flowlines import FlowlineTiler, FlowlineTilerPoly, Mapping
from lineworld.util.rastertools import normalize_to_uint8
from lineworld.util.slope import get_slope


@dataclass
class BathymetryFlowlinesMapLines:
    id: int | None
    lines: MultiLineString

    def todict(self) -> dict[str, int | float | str | None]:
        return {"lines": str(from_shape(self.lines))}


class BathymetryFlowlines(Layer):
    DATA_URL = ""
    DATA_SRID = Projection.WGS84

    DEFAULT_LAYER_NAME = "BathymetryFlowlines"

    def __init__(
        self,
        layer_id: str,
        db: engine.Engine,
        config: dict[str, Any] = {},
        tile_boundaries: list[Polygon] = [],
    ) -> None:
        super().__init__(layer_id, db, config)

        self.tile_boundaries = tile_boundaries

        self.data_dir = Path(
            Layer.DATA_DIR_NAME,
            self.config.get("layer_name", self.DEFAULT_LAYER_NAME).lower(),
        )
        self.source_file = Path(
            Layer.DATA_DIR_NAME, "elevation", "gebco_mosaic.tif"
        )  # TODO: hardcoded reference to elevation layer
        self.elevation_file = Path(
            self.data_dir, "flowlines_elevation.tif"
        )  # TODO: hardcoded reference to image file rendered by blender
        # self.highlights_file = Path(
        #     self.data_dir, "flowlines_highlights.png"
        # )  # TODO: hardcoded reference to image file rendered by blender
        self.density_file = Path("blender", "output.png")

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)

        metadata = MetaData()

        self.map_lines_table = Table(
            f"{self.config_name}_bathymetryflowlines_map_lines",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("lines", geoalchemy2.Geometry("LINESTRING"), nullable=False),
        )

        metadata.create_all(self.db)

    def extract(self) -> None:
        pass

    def transform_to_world(self) -> None:
        pass

    def transform_to_map(self, document_info: DocumentInfo) -> None:
        logger.info("reprojecting GeoTiff")

        with rasterio.open(self.source_file) as src:
            dst_crs = f"{document_info.projection.value[0]}:{document_info.projection.value[1]}"

            # calculate optimal output dimensions to get the width/height-ratio after reprojection
            _, dst_width, dst_height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            ratio = dst_width / dst_height

            px_per_mm = self.config.get("px_per_mm", 10.0)
            dst_width = int(document_info.width * px_per_mm)
            dst_height = int(document_info.width * px_per_mm * ratio)

            skip = False
            if self.elevation_file.exists():
                with rasterio.open(self.elevation_file) as dst:
                    if dst.width == dst_width and dst.height == dst_height:
                        skip = True
                        logger.info("reprojected GeoTiff exists, skipping")

            if not skip:
                xmin, ymin, xmax, ymax = transform_bounds(src.crs, dst_crs, *src.bounds)
                dst_transform = rasterio.transform.Affine(
                    (xmax - xmin) / float(dst_width),
                    0,
                    xmin,
                    0,
                    (ymin - ymax) / float(dst_height),
                    ymax,
                )

                kwargs = src.meta.copy()
                kwargs.update(
                    {
                        "crs": dst_crs,
                        "transform": dst_transform,
                        "width": dst_width,
                        "height": dst_height,
                    }
                )

                with rasterio.open(self.elevation_file, "w", **kwargs) as dst:
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
                            resampling=Resampling.nearest,
                        )

                logger.debug(f"reprojected fo file {self.elevation_file} | {dst_width} x {dst_height}px")

    def transform_to_lines(self, document_info: DocumentInfo) -> list[BathymetryFlowlinesMapLines]:
        flow_config = flowlines.FlowlineHatcherConfig()
        flow_config = lineworld.apply_config_to_object(self.config, flow_config)

        elevation = None
        with rasterio.open(self.elevation_file) as dataset:
            elevation = dataset.read(1)

        elevation = cv2.resize(elevation, [document_info.width, document_info.width])  # TODO

        density = None
        try:
            # use uint8 for the density map to save some memory and 256 values will be enough precision
            density = cv2.imread(str(self.density_file), cv2.IMREAD_GRAYSCALE)
            density = normalize_to_uint8(density)
            density = cv2.resize(density, [elevation.shape[1], elevation.shape[0]])

            # 50:50 blend of elevation data and externally computed density image
            elevation_normalized = normalize_to_uint8(elevation)
            # density = np.mean(np.dstack([density, elevation_normalized]), axis=2).astype(np.uint8)


        except Exception as e:
            logger.error(e)

        # MAX_TPI = 1_000
        # tpi = _calculate_topographic_position_index(data, 401)
        # tpi = np.clip(np.abs(tpi), 0, MAX_TPI)
        # normalized_tpi = normalize_to_uint8(tpi)

        _, _, _, _, angles, inclination = get_slope(elevation, 1)

        WINDOW_SIZE = 25
        MAX_WIN_VAR = 40000
        win_mean = ndimage.uniform_filter(elevation.astype(float), (WINDOW_SIZE, WINDOW_SIZE))
        win_sqr_mean = ndimage.uniform_filter(elevation.astype(float) ** 2, (WINDOW_SIZE, WINDOW_SIZE))
        win_var = win_sqr_mean - win_mean**2
        win_var = np.clip(win_var, 0, MAX_WIN_VAR)
        win_var = win_var * -1 + MAX_WIN_VAR
        win_var = (np.iinfo(np.uint8).max * ((win_var - np.min(win_var)) / np.ptp(win_var))).astype(np.uint8)

        # uint8 image must be centered around 128 to deal with negative values
        mapping_angle = ((angles + math.pi) / math.tau * 255.0).astype(np.uint8)

        mapping_flat = np.zeros_like(inclination, dtype=np.uint8)
        mapping_flat[inclination < flow_config.MIN_INCLINATION] = 255  # uint8

        mapping_distance = density  # uint8

        mapping_line_max_length = win_var  # uint8

        if self.config.get("blur_distance", False):
            kernel_size = self.config.get("blur_distance_kernel_size", 10)
            mapping_distance = cv2.blur(mapping_distance, (kernel_size, kernel_size))

        if self.config.get("blur_angles", False):
            kernel_size = self.config.get("blur_angles_kernel_size", 10)
            mapping_angle = cv2.blur(mapping_angle, (kernel_size, kernel_size))

        if self.config.get("blur_length", False):
            kernel_size = self.config.get("blur_length_kernel_size", 10)
            mapping_line_max_length = cv2.blur(mapping_line_max_length, (kernel_size, kernel_size))

        mappings = {
            Mapping.DISTANCE: mapping_distance,
            Mapping.ANGLE: mapping_angle,
            Mapping.MAX_LENGTH: mapping_line_max_length,
            Mapping.FLAT: mapping_flat,
        }

        tiler = None
        if self.tile_boundaries is not None and len(self.tile_boundaries) > 0:
            # convert from map coordinates to raster pixel coordinates
            mat_map_to_raster = document_info.get_transformation_matrix_map_to_raster(
                elevation.shape[1], elevation.shape[0]
            )
            raster_tile_boundaries = [
                affine_transform(boundary, mat_map_to_raster) for boundary in self.tile_boundaries
            ]

            tiler = FlowlineTilerPoly(
                mappings,
                flow_config,
                self.tile_boundaries, # map space
                raster_tile_boundaries, # raster space
                use_rust=True
            )
        else:
            tiler = FlowlineTiler(
                mappings,
                flow_config,
                (self.config.get("num_tiles", 4), self.config.get("num_tiles", 4)),
            )

        linestrings = tiler.hatch()

        # convert from raster pixel coordinates to map coordinates
        # mat_raster_to_map = document_info.get_transformation_matrix_raster_to_map(
        #     elevation.shape[1], elevation.shape[0]
        # )
        # linestrings = [affine_transform(line, mat_raster_to_map) for line in linestrings]
        linestrings = [line.simplify(self.config.get("tolerance", 0.1)) for line in linestrings]

        # TODO: this should be a function in geometrytools
        linestrings_filtered = []
        for g in linestrings:
            match g:
                case LineString():
                    linestrings_filtered.append(g)
                case GeometryCollection():
                    for sg in g.geoms:
                        if type(sg) is LineString:
                            linestrings_filtered.append(sg)
                case _:
                    logger.warning(f"unexpected geometry type during filtering: {type(g)}")

        return [BathymetryFlowlinesMapLines(None, line) for line in linestrings_filtered]

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

    def out(
        self, exclusion_zones: list[Polygon], document_info: DocumentInfo
    ) -> tuple[list[shapely.Geometry], list[Polygon]]:
        """
        Returns (drawing geometries, exclusion polygons)
        """

        drawing_geometries = []
        with self.db.begin() as conn:
            result = conn.execute(select(self.map_lines_table))
            drawing_geometries = [to_shape(row.lines) for row in result]

        # cut extrusion_zones into drawing_geometries
        # Note: using a STRtree here instead of unary_union() and difference() is a 6x speedup
        drawing_geometries_cut = []
        tree = STRtree(exclusion_zones)

        viewport = document_info.get_viewport()

        for g in drawing_geometries:
            g_processed = shapely.intersection(g, viewport)
            if g_processed.is_empty:
                continue

            for i in tree.query(g):
                g_processed = shapely.difference(g_processed, exclusion_zones[i])
                if g_processed.is_empty:
                    break
            else:
                drawing_geometries_cut.append(g_processed)

        # and do not add anything to exclusion_zones
        return (drawing_geometries_cut, exclusion_zones)
