import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import geoalchemy2
import numpy as np
import rasterio
import shapely
from core.maptools import DocumentInfo, Projection
from geoalchemy2.shape import to_shape, from_shape
from layers.layer import Layer
from loguru import logger
from scipy import ndimage
from shapely import Polygon, MultiLineString, STRtree
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
from experiments.hatching import flowlines
from experiments.hatching.flowlines import FlowlineTiler, FlowlineTilerPoly
from lineworld.util.gebco_grid_to_polygon import _calculate_topographic_position_index
from lineworld.util.rastertools import normalize_to_uint8


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
        self.highlights_file = Path(
            self.data_dir, "flowlines_highlights.png"
        )  # TODO: hardcoded reference to image file rendered by blender
        self.density_file = Path("blender", "output.png")

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)

        metadata = MetaData()

        self.map_lines_table = Table(
            "bathymetryflowlines_map_lines",
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

        data = None
        with rasterio.open(self.elevation_file) as dataset:
            data = dataset.read(1)

        data = cv2.resize(data, [document_info.width * 10, document_info.width * 10])  # TODO

        flow_config.MM_TO_PX_CONVERSION_FACTOR = data.shape[1] / document_info.width

        density = None
        try:
            # use uint8 for the density map to save some memory and 256 values will be enough precision
            density = cv2.imread(str(self.density_file), cv2.IMREAD_GRAYSCALE)
            density = normalize_to_uint8(density)
            density = cv2.resize(density, data.shape)

            # 50:50 blend of elevation data and externally computed density image
            data_normalized = normalize_to_uint8(data)

            density = np.mean(np.dstack([density, data_normalized]), axis=2).astype(np.uint8)

        except Exception as e:
            logger.error(e)

        # MAX_TPI = 1_000
        # tpi = _calculate_topographic_position_index(data, 401)
        # tpi = np.clip(np.abs(tpi), 0, MAX_TPI)
        # normalized_tpi = normalize_to_uint8(tpi)

        WINDOW_SIZE = 25
        MAX_WIN_VAR = 40000
        win_mean = ndimage.uniform_filter(data.astype(float), (WINDOW_SIZE, WINDOW_SIZE))
        win_sqr_mean = ndimage.uniform_filter(data.astype(float) ** 2, (WINDOW_SIZE, WINDOW_SIZE))
        win_var = win_sqr_mean - win_mean**2
        win_var = np.clip(win_var, 0, MAX_WIN_VAR)
        win_var = win_var * -1 + MAX_WIN_VAR
        win_var = (np.iinfo(np.uint8).max * ((win_var - np.min(win_var)) / np.ptp(win_var))).astype(np.uint8)

        mappings = np.zeros([data.shape[0], data.shape[1], 2], dtype=np.uint8)
        mappings[:, :, flowlines.MAPPING_DISTANCE] = density[:, :]
        mappings[:, :, flowlines.MAPPING_MAX_SEGMENTS] = win_var[:, :]

        tiler = None
        if self.tile_boundaries is not None and len(self.tile_boundaries) > 0:
            # convert from map coordinates to raster pixel coordinates
            mat = document_info.get_transformation_matrix_map_to_raster(data.shape[1], data.shape[0])
            raster_tile_boundaries = [affine_transform(boundary, mat) for boundary in self.tile_boundaries]

            tiler = FlowlineTilerPoly(data, mappings, flow_config, raster_tile_boundaries)
        else:
            tiler = FlowlineTiler(
                data,
                mappings,
                flow_config,
                (self.config.get("num_tiles", 4), self.config.get("num_tiles", 4)),
            )

        linestrings = tiler.hatch()

        # convert from raster pixel coordinates to map coordinates
        mat = document_info.get_transformation_matrix_raster_to_map(data.shape[1], data.shape[0])
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
        for g in drawing_geometries:
            g_processed = g
            for i in tree.query(g):
                g_processed = shapely.difference(g_processed, exclusion_zones[i])
                if g_processed.is_empty:
                    break
            else:
                drawing_geometries_cut.append(g_processed)

        # and do not add anything to exclusion_zones
        return (drawing_geometries_cut, exclusion_zones)
