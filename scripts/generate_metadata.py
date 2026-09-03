"""
Baut eine zentrale meta.json aus allen erzeugten PNGs in output/.

Erwartetes Dateinamensschema (wie von den Rendering-Skripten erzeugt):
    {var_type}_{YYYYMMDD}_{HHMM}.wepb

var_type darf selbst Unterstriche enthalten (z.B. "tp_acc", "dbz_cmax",
"change_snow") - das Skript erkennt das Zeitstempel-Suffix per Regex und
nimmt alles davor als var_type.

Aufruf:
    python build_meta.py output
    python build_meta.py output --run 18 --date 20260820

--run / --date sind optional. Wenn sie fehlen (z.B. beim lokalen Testen),
wird "date" aus dem fruehesten gefundenen Timestep abgeleitet und "run"
bleibt leer - die GitHub-Actions-Workflow-Datei kann sie spaeter einfach
mitgeben.

Keine externen Abhaengigkeiten - laeuft mit reiner Python-Standardbibliothek
(math statt numpy), damit dieser Schritt in der CI kein pip install braucht.
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

FILENAME_RE = re.compile(r"^(?P<var_type>.+)_(?P<date>\d{8})_(?P<time>\d{4})\.webp$")

# ------------------------------
# Bounding Box: volle ICON-D2-Modelldomain (fix)
# ------------------------------
# WICHTIG: dieser Wert MUSS exakt mit "extent" im Rendering-Skript
# uebereinstimmen! Ein PNG traegt selbst keine Geo-Info - die Position
# auf der Karte wird einzig durch den imageExtent bestimmt. Damit das
# nicht auseinanderlaufen kann, wird der Wert hier zusaetzlich in die
# meta.json geschrieben; das HTML kann ihn dann von dort lesen statt
# ihn separat hart zu codieren.
EXTENT_LONLAT = [-3.94, 20.34, 43.18, 58.08]  # lon_min, lon_max, lat_min, lat_max

EARTH_RADIUS = 6378137.0  # Meter, WGS84/Web-Mercator-Kugelradius


def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * math.radians(lon_deg)
    y = EARTH_RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2))
    return x, y


def compute_extent_3857(extent_lonlat):
    lon_min, lon_max, lat_min, lat_max = extent_lonlat
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def scan_output_dir(output_dir: str):
    """Durchsucht output_dir (inkl. evtl. Unterordner je var_type) nach
    PNGs und gruppiert die Timesteps pro var_type."""
    var_types: dict[str, set[str]] = {}

    for root, _dirs, files in os.walk(output_dir):
        for fname in files:
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            var_type = m.group("var_type")
            ts = f"{m.group('date')}_{m.group('time')}"
            var_types.setdefault(var_type, set()).add(ts)

    return var_types


def build_meta(output_dir: str, run: str | None, date: str | None):
    var_types_raw = scan_output_dir(output_dir)

    if not var_types_raw:
        print(f"Keine passenden PNGs in {output_dir} gefunden.", file=sys.stderr)

    var_types_out = {}
    all_timesteps: list[str] = []

    for var_type in sorted(var_types_raw):
        timesteps = sorted(var_types_raw[var_type])
        all_timesteps.extend(timesteps)
        var_types_out[var_type] = {
            "num_steps": len(timesteps),
            "timesteps": timesteps,
        }

    # date automatisch aus dem fruehesten Timestep ableiten, falls nicht
    # explizit uebergeben (z.B. spaeter via --date aus der GitHub-Action)
    if not date:
        date = min(all_timesteps).split("_")[0] if all_timesteps else ""

    meta = {
        "run": run or "",
        "date": date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "crs": "EPSG:3857",
        "extent": EXTENT_LONLAT,
        "extent_3857": compute_extent_3857(EXTENT_LONLAT),
        "var_types": var_types_out,
    }
    return meta


def main():
    png_root = sys.argv[1]
    run = sys.argv[2]
    date = sys.argv[3] if len(sys.argv) > 3 else datetime.now(timezone.utc).strftime("%Y%m%d")

    meta = build_meta(png_root, run, date)

    # metadata.json liegt außerhalb des run-Ordners
    meta_path = os.path.join(os.path.dirname(png_root), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Metadata written to {meta_path}")

    total_vars = len(meta["var_types"])
    total_files = sum(v["num_steps"] for v in meta["var_types"].values())
    print(f"meta.json geschrieben: {meta_path} ({total_vars} var_types, {total_files} Dateien)")


if __name__ == "__main__":
    main()
