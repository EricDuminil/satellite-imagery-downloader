import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from rich.progress import Progress

# TODO: Add cache? Some tiles appear to be black, and should be redownloaded
# TODO: Add warning / errors if too many tiles?
# NOTE: Could possibly use rich.progress to display progress

MAX_THREADS = 8
WAIT_BETWEEN_DOWNLOADS = 3


def random_wait(mean_time: int = WAIT_BETWEEN_DOWNLOADS, variation: int = 2) -> None:
    random_time = max([1, mean_time + random.randint(-variation, variation)])
    time.sleep(random_time)


def download_tile(url, headers, channels):
    random_wait()
    # print(f"Download {url}...")
    response = requests.get(url, headers=headers)
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)

    if channels == 3:
        return cv2.imdecode(arr, 1)
    return cv2.imdecode(arr, -1)


# Mercator projection
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def tile_to_quadkey(x: int, y: int, z: int) -> str:
    """
    Converts a Bing Maps tile's X, Y coordinates and Zoom level Z into its Quadkey. {q} can then be used in url format.

    see https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
    """

    quadkey = ""

    for i in range(z - 1, -1, -1):
        y_bit = (y >> i) & 1
        x_bit = (x >> i) & 1

        quadkey_digit = (2 * y_bit) + x_bit

        quadkey += str(quadkey_digit)

    return quadkey


def download_image(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    zoom: int,
    url: str,
    headers: dict,
    tile_size: int = 256,
    channels: int = 3,
) -> np.ndarray:
    """
    Downloads a map region. Returns an image stored as a `numpy.ndarray` in BGR or BGRA, depending on the number
    of `channels`.

    Parameters
    ----------
    `(lat1, lon1)` - Coordinates (decimal degrees) of the top-left corner of a rectangular area

    `(lat2, lon2)` - Coordinates (decimal degrees) of the bottom-right corner of a rectangular area

    `zoom` - Zoom level

    `url` - Tile URL with {x}, {y} and {z} in place of its coordinate and zoom values

    `headers` - Dictionary of HTTP headers

    `tile_size` - Tile size in pixels

    `channels` - Number of channels in the output image. Also affects how the tiles are converted into numpy arrays.
    """

    scale = 1 << zoom

    # Find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    columns_count = br_tile_x + 1 - tl_tile_x
    rows_count = br_tile_y + 1 - tl_tile_y

    with Progress() as progress:
        total_task = progress.add_task("[green]Downloading....", total=columns_count * rows_count)
        def build_row(tile_y):
            current_task = progress.add_task(f"[blue]Row {tile_y}....", total=columns_count)
            for tile_x in range(tl_tile_x, br_tile_x + 1):
                tile = download_tile(
                    url.format(x=tile_x, y=tile_y, z=zoom, q=tile_to_quadkey(tile_x, tile_y, zoom)), headers, channels
                )

                if tile is not None:
                    # Find the pixel coordinates of the new tile relative to the image
                    tl_rel_x = tile_x * tile_size - tl_pixel_x
                    tl_rel_y = tile_y * tile_size - tl_pixel_y
                    br_rel_x = tl_rel_x + tile_size
                    br_rel_y = tl_rel_y + tile_size

                    # Define where the tile will be placed on the image
                    img_x_l = max(0, tl_rel_x)
                    img_x_r = min(img_w + 1, br_rel_x)
                    img_y_l = max(0, tl_rel_y)
                    img_y_r = min(img_h + 1, br_rel_y)

                    # Define how border tiles will be cropped
                    cr_x_l = max(0, -tl_rel_x)
                    cr_x_r = tile_size + min(0, img_w - br_rel_x)
                    cr_y_l = max(0, -tl_rel_y)
                    cr_y_r = tile_size + min(0, img_h - br_rel_y)

                    img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]
                    progress.update(total_task, advance=1)
                    progress.update(current_task, advance=1)
            progress.remove_task(current_task)

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            for tile_y in range(tl_tile_y, br_tile_y + 1):
                executor.submit(build_row, tile_y)

            executor.shutdown(wait=True)

    return img


def image_size(lat1: float, lon1: float, lat2: float, lon2: float, zoom: int, tile_size: int = 256):
    """Calculates the size of an image without downloading it. Returns the width and height in pixels as a tuple."""

    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y
