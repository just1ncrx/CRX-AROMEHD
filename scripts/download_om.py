#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BUCKET_URL = "https://openmeteo.s3.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

MODEL = "meteofrance_arome_france_hd_15min"
OUTDIR = Path("data/parameter")

# --- Konfiguration aus Umgebungsvariablen ---------------------------------
YEAR = os.environ.get("YEAR")
MONTH = os.environ.get("MONTH")
DAY = os.environ.get("DAY")
RUN = os.environ.get("RUN")
WORKERS = int(os.environ.get("WORKERS", "4"))


def list_objects(prefix: str) -> list[str]:
    """Listet alle Objekt-Keys unter einem Prefix via S3 ListObjectsV2 API
    (mit Pagination, da der Bucket public ist)."""
    keys = []
    continuation_token = None

    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation_token:
            params["continuation-token"] = continuation_token

        resp = requests.get(BUCKET_URL, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        for contents in root.findall(f"{S3_NS}Contents"):
            key = contents.find(f"{S3_NS}Key").text
            keys.append(key)

        truncated = root.find(f"{S3_NS}IsTruncated")
        if truncated is not None and truncated.text == "true":
            token_el = root.find(f"{S3_NS}NextContinuationToken")
            if token_el is None:
                break
            continuation_token = token_el.text
        else:
            break

    return keys


def download_file(key: str) -> str:
    url = f"{BUCKET_URL}/{key}"
    dest = OUTDIR / Path(key).name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return f"uebersprungen (existiert bereits): {dest}"

    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return f"heruntergeladen: {dest}"


def main():
    missing = [name for name, val in
               (("YEAR", YEAR), ("MONTH", MONTH), ("DAY", DAY), ("RUN", RUN))
               if not val]
    if missing:
        sys.exit(f"Fehlende Umgebungsvariablen: {', '.join(missing)}")

    run = str(RUN).zfill(2)
    month = str(MONTH).zfill(2)
    day = str(DAY).zfill(2)
    prefix = f"data_spatial/{MODEL}/{YEAR}/{month}/{day}/{run}00Z/"

    print(f"Prefix:  {prefix}")
    print(f"Outdir:  {OUTDIR}")
    print(f"Workers: {WORKERS}")

    keys = list_objects(prefix)
    if not keys:
        sys.exit(f"Keine Dateien unter diesem Prefix gefunden: {prefix}")

    keys = [k for k in keys if k.endswith(".om")]
    if not keys:
        sys.exit(f"Keine .om-Dateien unter diesem Prefix gefunden: {prefix}")

    print(f"{len(keys)} .om-Datei(en) gefunden")

    errors = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_file, key): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            try:
                print(future.result())
            except Exception as e:
                print(f"FEHLER bei {key}: {e}")
                errors.append(key)

    if errors:
        sys.exit(f"{len(errors)} Datei(en) fehlgeschlagen: {errors}")

    print("Fertig.")


if __name__ == "__main__":
    main()
