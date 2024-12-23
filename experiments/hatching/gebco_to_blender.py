from contextlib import ExitStack
from pathlib import Path

import rasterio
from loguru import logger
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

DATA_URL = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2024/geotiff/"

DATA_DIR = Path("experiments/hatching/data", "GebcoToBlender".lower())
TILES_DIR = Path(DATA_DIR, "tiles")
SCALED_DIR = Path(DATA_DIR, "scaled")
MOSAIC_FILE = Path(DATA_DIR, "gebco_mosaic.tif")
REPROJECT_FILE = Path(DATA_DIR, "reproject.tif")

GEOTIFF_SCALING_FACTOR = 1
# GEOTIFF_SCALING_FACTOR = 1/8 # correct ratio for blender


def downscale_and_write(input_path: Path, output_path: Path, scaling_factor: float) -> None:
    """
    Downscale GEBCO GeoTiff images
    """

    with rasterio.open(input_path) as src:
        data = src.read(
            out_shape=(src.count, int(src.height * scaling_factor), int(src.width * scaling_factor)),
            resampling=Resampling.bilinear
        )

        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )

        config = {
            "driver": "GTiff",
            "height": data.shape[-2],
            "width": data.shape[-1],
            "count": 1,
            "dtype": data.dtype,
            "crs": src.crs,
            "transform": transform
        }

        with rasterio.open(output_path, "w", **config) as dst:
            dst.write(data)


def merge_and_write(geotiff_paths: list[Path], output_path: Path) -> None:
    with ExitStack() as stack:
        tiles = [stack.enter_context(rasterio.open(geotiff_path)) for geotiff_path in geotiff_paths]

        mosaic, mosaic_transform = merge(tiles, resampling=Resampling.bilinear)

        config = {
            "driver": "GTiff",
            "height": mosaic.shape[-2],
            "width": mosaic.shape[-1],
            "count": 1,
            "dtype": mosaic.dtype,
            "crs": tiles[0].crs,
            "transform": mosaic_transform
        }

        with rasterio.open(output_path, "w", **config) as dst:
            dst.write(mosaic)


def reproject_dataset(src: Path, dst: Path) -> None:
    dst_crs = "ESRI:54029"

    with rasterio.open(src) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                # remove any above-waterlevel terrain
                band = rasterio.band(src, i)
                band_arr = src.read(i)
                band_arr[band_arr > 0] = 0

                reproject(
                    source=band_arr,
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )


if __name__ == "__main__":

    logger.info("extracting elevation data from GeoTiffs")

    # Downscaling
    dataset_files = [f for f in TILES_DIR.iterdir() if f.is_file() and f.suffix == ".tif"]
    if len(dataset_files) == 0:
        logger.warning("no GeoTiffs to transform")

    scaled_files = []
    for dataset_file in dataset_files:
        scaled_path = Path(SCALED_DIR, dataset_file.name)
        scaled_files.append(scaled_path)

        # if scaled_path.exists():
        #     continue

        logger.debug(f"downscaling tile: {dataset_file}")
        downscale_and_write(dataset_file, scaled_path, GEOTIFF_SCALING_FACTOR)

    # Merging tiles into a mosaic
    # if not MOSAIC_FILE.exists():
    logger.debug("merging mosaic tiles")
    merge_and_write(scaled_files, MOSAIC_FILE)

    # Reprojecting
    # if not REPROJECT_FILE.exists():
    logger.debug("reprojecting mosaic")
    reproject_dataset(MOSAIC_FILE, REPROJECT_FILE)
