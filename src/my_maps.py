from pathlib import Path

import cv2

from image_downloading import download_image

SCRIPT_DIR = Path(__file__).resolve().parent

MAPS = {
    "Groix": [47.66480191718762, -3.5404115924138733, 47.60663695544539, -3.400617087564814, 17],
    "Houat": [47.413654718057984, -3.0214163590517242, 47.36368529994759, -2.9306483417315765, 18],
    "Belle-Ile": [47.407899134037365, -3.289912020518811, 47.26545760031402, -3.0448569371450986, 16],
    "Hoedic": [47.35996326489376, -2.910067939653619, 47.31106708683253, -2.8246144865125857, 18],
    "Les Glénans": [47.746850565803975, -4.063520659844032, 47.683716225683675, -3.9352222847223755, 17],
}

HEADERS = {
    "cache-control": "max-age=0",
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36",
}

BING_URL = "http://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
GOOGLE_URL = "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

OUTPUT_FOLDER = SCRIPT_DIR / "images"

OUTPUT_FOLDER.mkdir(exist_ok=True)

for name, (lat1, lon1, lat2, lon2, zoom) in MAPS.items():
    output_map = OUTPUT_FOLDER / f"{name}_{zoom}.png"
    if output_map.exists() > 0:
        continue
    print(f"Downloading {output_map}")
    img = download_image(lat1, lon1, lat2, lon2, zoom, BING_URL, HEADERS)
    cv2.imwrite(output_map.as_posix(), img)
