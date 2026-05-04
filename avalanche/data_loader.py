from __future__ import annotations

from typing import Any


def extract_dem_tiles(tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dem_tiles = []
    for tile in tiles:
        dem_tiles.append({"tile_id": tile["tile_id"], "dem": tile["dem"], "meta": tile.get("meta", {})})
    return dem_tiles
