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
from flowlines_py import flowlines_py

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
import netCDF4

import lineworld
from lineworld.core import flowlines
from lineworld.core.flowlines import FlowlineTiler, FlowlineTilerPolyRust, Mapping, _py_config_to_rust_config
from lineworld.util.rastertools import normalize_to_uint8
from lineworld.util.slope import get_slope


@dataclass
class OceanCurrentsMapLines:
    id: int | None
    lines: MultiLineString

    def todict(self) -> dict[str, int | float | str | None]:
        return {"lines": str(from_shape(self.lines))}


class OceanCurrents(Layer):
    DATA_URL = ""
    DATA_SRID = Projection.WGS84

    DEFAULT_LAYER_NAME = "OceanCurrents"

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

        self.source_file = Path(self.data_dir, "oscar_currents_final_20220101.nc")  # TODO: hardcoded reference

        self.reprojection_file = Path(self.data_dir, "reproject.tif")

        if not self.data_dir.exists():
            os.makedirs(self.data_dir)

        metadata = MetaData()

        self.map_lines_table = Table(
            f"{self.config_name}_oceancurrents_map_lines",
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

        with netCDF4.Dataset(self.source_file, "r", format="NETCDF4") as data:
            # convert xarray to numpy NdArray, fill empty pixels with zeros
            u = data.variables["u"][:].filled(0)[0, :, :]
            v = data.variables["v"][:].filled(0)[0, :, :]

            # swap lat lon axes
            u = u.T
            v = v.T

            bands = [u, v]
            src_height, src_width = u.shape

            # Resolution: 0.25°, decimal degrees per pixel
            src_resolution = 0.25

            # Target resolution, in units of target coordinate reference system
            dst_resolution = 10_000.0

            dst_crs = str(document_info.projection)

            src_transform = rasterio.Affine.translation(0, -90) * rasterio.Affine.scale(src_resolution, src_resolution)
            src_crs = {"init": str(self.DATA_SRID)}  # rasterio-style CRS dict

            # Origin: top-left. Order: left, bottom, right, top
            src_bounds = [-180, -90, 180, 90]
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height, *src_bounds, resolution=dst_resolution
            )

            params = {
                "width": dst_width,
                "height": dst_height,
                "count": len(bands),
                "crs": dst_crs,
                "transform": dst_transform,
                "dtype": np.float32,
            }

            with rasterio.open(self.reprojection_file, "w", **params) as dst:
                for i, band in enumerate(bands):
                    reproject(
                        source=band,
                        destination=rasterio.band(dst, i + 1),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )

    def transform_to_lines(self, document_info: DocumentInfo) -> list[OceanCurrentsMapLines]:
        flow_config = flowlines.FlowlineHatcherConfig()
        flow_config = lineworld.apply_config_to_object(self.config, flow_config)

        with rasterio.open(self.reprojection_file) as dataset:
            u = dataset.read(1)
            v = dataset.read(2)

            angles = np.arctan2(u, v)
            magnitude = np.hypot(u, v)

            angles = (angles + math.pi / 2) % math.tau

            # center around math.pi (128) so we avoid negative values
            mapping_angle = angles + math.pi
            mapping_angle = ((mapping_angle / math.tau) * 255).astype(np.uint8)

            mapping_non_flat = np.zeros_like(magnitude, dtype=np.uint8)
            mapping_non_flat[magnitude > flow_config.MIN_INCLINATION] = 255  # uint8

            magnitude = np.clip(magnitude, 0, 1)
            mapping_distance = ~normalize_to_uint8(magnitude)  # uint8

            mapping_line_max_length = np.full_like(mapping_angle, 255)

            mappings = {
                Mapping.DISTANCE: mapping_distance,
                Mapping.ANGLE: mapping_angle,
                Mapping.MAX_LENGTH: mapping_line_max_length,
                Mapping.NON_FLAT: mapping_non_flat,
            }

            mat_map_to_raster = document_info.get_transformation_matrix_map_to_raster(u.shape[1], u.shape[0])
            raster_tile_boundaries = [
                affine_transform(boundary, mat_map_to_raster) for boundary in self.tile_boundaries
            ]

            # tiler = FlowlineTilerPoly(mappings, flow_config, raster_tile_boundaries)
            tiler = FlowlineTilerPolyRust(mappings, flow_config, raster_tile_boundaries)
            linestrings = tiler.hatch()

            # no tiling, compute all at once
            # rust_config = flowlines_py.FlowlinesConfig()
            # rust_config = _py_config_to_rust_config(flow_config, rust_config)
            # mappings = [mapping_distance, mapping_angle, mapping_line_max_length, mapping_non_flat]
            # rust_lines: list[list[tuple[float, float]]] = flowlines_py.hatch([u.shape[1], u.shape[0]], rust_config, *mappings)
            # linestrings = [LineString(l) for l in rust_lines]

            # convert from raster pixel coordinates to map coordinates
            mat_raster_to_map = document_info.get_transformation_matrix_raster_to_map(u.shape[1], u.shape[0])
            linestrings = [affine_transform(line, mat_raster_to_map) for line in linestrings]
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

            return [OceanCurrentsMapLines(None, line) for line in linestrings_filtered]

    def load(self, geometries: list[OceanCurrentsMapLines]) -> None:
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
