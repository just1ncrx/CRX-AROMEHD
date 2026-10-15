import sys
import os
import re
import gc
import json
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from omfiles import OmFileReader

from encoding_config import NODATA_U8, LEVELS, ENCODING

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "data/parameter"
output_dir = sys.argv[2]      # z.B. "aromehd/t2m"
var_type = sys.argv[3]        # 't2m', 'wind', 'tp15', 'cape_ml', 'wwhd'
os.makedirs(output_dir, exist_ok=True)

if var_type not in ENCODING:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)
enc = ENCODING[var_type]

# ------------------------------
# Zeitschrittlaenge des Modells in Sekunden (15min-Modell).
# ------------------------------
DT_SECONDS = 900

# ------------------------------
# var_type -> Name des Kindes (Variable) INNERHALB jeder .om Datei.
# Im data_spatial-Layout enthaelt JEDE Datei ALLE Variablen fuer genau
# einen Zeitschritt (root ist eine Gruppe, kein Array). Der Dateiname
# selbst ist der Zeitstempel, z.B. "2026-09-03T1900.om".
# "wwhd" ist ein Sonderfall (liest zwei Kinder) und steht deshalb nicht hier.
# ------------------------------
OM_CHILD_NAMES = {
    "t2m": "temperature_2m",
    "wind": "wind_gusts_10m",
    "tp15": "precipitation",
    "cape_ml": "cape",
}

# Regex fuer den Zeitstempel im Dateinamen: YYYY-MM-DDTHHMM.om
FILENAME_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{4})")

if var_type != "wwhd":
    child_name = OM_CHILD_NAMES.get(var_type)
    if child_name is None:
        print(f"var_type '{var_type}' hat noch kein Kind-Name-Mapping in OM_CHILD_NAMES")
        sys.exit(1)

# Umrechnung Rohwert -> Anzeige-Einheit, je Variable
UNIT_CONVERT = {
    "t2m": lambda v: v - 273.15 if np.nanmax(v) > 200 else v,  # K -> °C, nur falls noetig
    "wind": lambda v: v * 3.6,  # m/s -> km/h
}
convert = UNIT_CONVERT.get(var_type, lambda v: v)

# Interpolation: Wettercodes duerfen nie gemittelt werden -> nearest
INTERP_METHOD = "nearest" if var_type == "wwhd" else "linear"

# ------------------------------
# Globale Bbox der Quelldateien (AROME-HD, bestaetigt via crs_wkt-Kind der .om Datei)
# ------------------------------
GLOBAL_LAT_MIN, GLOBAL_LON_MIN = 37.5, -12.0
GLOBAL_LAT_MAX, GLOBAL_LON_MAX = 55.4, 16.0

# Speicherreihenfolge der Lat-Achse in der Datei (Zeile 0 = Norden oder Sueden?).
# Verifiziert: Zeile 0 = Suedpol/Sueden -> aufsteigend gespeichert.
LAT_STORED_DESCENDING = False

# Dimensionsreihenfolge im data_spatial-Layout: jede Variable ist ein 2D-Array
# (lat, lon) OHNE eigene Zeitachse (ein Zeitschritt pro Datei).

# ------------------------------
# Ausgaberaster (identisch zum ICON-D2-/IFS-Raster)
# ------------------------------
# regular-lat-lon, EPSG:4326, row0 = Norden, Werte = Pixelzentren.
# MUSS zu GRID in generate_metadata.py passen (dort wird die Dateigröße geprüft).
# Bereiche ausserhalb der AROME-Domaene (z.B. lon > 16, lat > 55.4) werden NoData (255).
GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX = -3.94, 43.18, 20.34, 58.08
GRID_WIDTH, GRID_HEIGHT = 1215, 746

# Ausschnitt, der aus dem globalen Gitter gelesen wird (= Ausgaberaster)
extent = [GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX]  # lon_min, lon_max, lat_min, lat_max

lon_t = np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_WIDTH)
lat_t = np.linspace(GRID_LAT_MAX, GRID_LAT_MIN, GRID_HEIGHT)   # Nord -> Süd
lat_tt, lon_tt = np.meshgrid(lat_t, lon_t, indexing="ij")      # Shape (HEIGHT, WIDTH)
target_points = np.column_stack((lat_tt.ravel(), lon_tt.ravel()))


def encode_uint8(data, enc):
    """float-Array (NaN erlaubt) -> uint8-Array (255 = NoData)."""
    nan_mask = ~np.isfinite(data)
    vmin, vmax, tr = enc["vmin"], enc["vmax"], enc["transform"]
    d = np.where(nan_mask, vmin, data).astype(np.float64)

    if tr == "raw":
        q = np.clip(np.rint(d), 0, LEVELS)
    else:
        d = np.clip(d, vmin, vmax)
        if tr == "linear":
            u = (d - vmin) / (vmax - vmin)
        elif tr == "sqrt":
            u = np.sqrt((d - vmin) / (vmax - vmin))
        elif tr == "ssqrt":
            u = (np.sign(d) * np.sqrt(np.abs(d) / vmax) + 1.0) / 2.0
        else:
            raise ValueError(tr)
        q = np.rint(u * LEVELS)

    q = q.astype(np.uint8)
    q[nan_mask] = NODATA_U8
    return q


def to_target_grid(data, lon, lat, method="linear"):
    """Reguläres Quellgitter (lat aufsteigend, lon aufsteigend) -> Ausgaberaster
    (HEIGHT, WIDTH), row0 = Norden. Außerhalb des Quellgitters = NaN."""
    interp = RegularGridInterpolator(
        (lat, lon), data, method=method, bounds_error=False, fill_value=np.nan
    )
    return interp(target_points).reshape(GRID_HEIGHT, GRID_WIDTH)


def compute_crop_indices(nlat, nlon):
    lat_res = (GLOBAL_LAT_MAX - GLOBAL_LAT_MIN) / (nlat - 1)
    lon_res = (GLOBAL_LON_MAX - GLOBAL_LON_MIN) / (nlon - 1)

    lon_min, lon_max, lat_min, lat_max = extent

    if LAT_STORED_DESCENDING:
        row_of = lambda v: (GLOBAL_LAT_MAX - v) / lat_res  # Zeile 0 = Norden
    else:
        row_of = lambda v: (v - GLOBAL_LAT_MIN) / lat_res  # Zeile 0 = Sueden

    r1, r2 = sorted([row_of(lat_min), row_of(lat_max)])
    row_start = max(0, int(np.floor(r1)) - 1)
    row_end = min(nlat - 1, int(np.ceil(r2)) + 1)

    col_of = lambda v: (v - GLOBAL_LON_MIN) / lon_res
    col_start = max(0, int(np.floor(col_of(lon_min))) - 1)
    col_end = min(nlon - 1, int(np.ceil(col_of(lon_max))) + 1)

    return row_start, row_end, col_start, col_end, lat_res, lon_res


def crop_lat_lon_arrays(row_start, row_end, col_start, col_end, lat_res, lon_res, nlat):
    if LAT_STORED_DESCENDING:
        lat_full = GLOBAL_LAT_MAX - np.arange(nlat) * lat_res
    else:
        lat_full = GLOBAL_LAT_MIN + np.arange(nlat) * lat_res
    lat_crop = lat_full[row_start:row_end + 1]
    lon_crop = GLOBAL_LON_MIN + np.arange(col_start, col_end + 1) * lon_res
    return lat_crop, lon_crop


def parse_valid_time_from_filename(filename):
    """Der Dateiname im data_spatial-Layout IST der Zeitstempel, z.B.
    '2026-09-03T1900.om' -> 2026-09-03 19:00 UTC. Jede Datei enthaelt
    genau diesen einen Zeitschritt fuer alle Variablen."""
    match = FILENAME_TIME_RE.search(filename)
    if not match:
        raise ValueError(
            f"{filename}: Zeitstempel nicht im Dateinamen gefunden "
            f"(erwartet Format YYYY-MM-DDTHHMM.om)"
        )
    return dt.datetime.strptime(match.group(1), "%Y-%m-%dT%H%M").replace(tzinfo=dt.timezone.utc)


# ------------------------------
# ww-Berechnung (Regen/Schneeregen/Schnee aus precip + snowfall_water_equivalent)
# ------------------------------
def calculate_ww(precip, snow_we, dt_seconds):
    """
    precip:  mm/Zeitschritt (Gesamt, Regen+Schnee als Wasseräquivalent)
    snow_we: mm WE/Zeitschritt (nur Schneeanteil)
    dt_seconds: Zeitschrittlänge des Modells in Sekunden

    Rückgabe: float-Array mit Codes 0, 51, 56/57, 61/63/65, 71/73/75, oder
    NaN dort, wo precip/snow_we selbst NaN war (= außerhalb des
    AROME-Modellgebiets, nicht "kein Niederschlag"!).
    """
    dt_h = dt_seconds / 900.0

    invalid = ~np.isfinite(precip) | ~np.isfinite(snow_we)

    precip_safe = np.where(invalid, 0.0, precip)
    snow_safe = np.where(invalid, 0.0, snow_we)

    rain_mm = np.clip(precip_safe - snow_safe, 0, None)   # reiner Regenanteil
    snow_cm = snow_safe * 0.7                              # mm(WE) -> cm grob

    rain_rate = rain_mm / dt_h   # mm/h
    snow_rate = snow_cm / dt_h   # cm/h

    has_precip = (precip_safe / dt_h) > 0.01
    is_snow  = has_precip & (snow_rate > 0) & (rain_rate <= 0.05)
    is_mixed = has_precip & (snow_rate > 0) & (rain_rate  > 0.05)
    is_rain  = has_precip & (rain_rate  > 0) & ~is_mixed

    code = np.zeros(precip.shape, dtype=np.float32)  # Default: 0

    # Schneefall-Intensität
    code = np.where(is_snow & (snow_rate < 0.2), 71, code)
    code = np.where(is_snow & (snow_rate >= 0.2) & (snow_rate < 0.8), 73, code)
    code = np.where(is_snow & (snow_rate >= 0.8), 75, code)

    # Schneeregen (eigene Definition, nicht offizielles WMO 56/57)
    code = np.where(is_mixed & (rain_rate < 2.5), 56, code)
    code = np.where(is_mixed & (rain_rate >= 2.5), 57, code)

    # Niesel-/Regen-Intensität
    code = np.where(is_rain & (rain_rate < 0.2), 51, code)
    code = np.where(is_rain & (rain_rate >= 0.2) & (rain_rate < 2.5), 61, code)
    code = np.where(is_rain & (rain_rate >= 2.5) & (rain_rate < 7.6), 63, code)
    code = np.where(is_rain & (rain_rate >= 7.6), 65, code)

    # Außerhalb des Modellgebiets -> NaN (wird später zu NoData 255)
    code[invalid] = np.nan

    return code


# ------------------------------
# Metadaten fuer die Konsolenausgabe (wie im IFS-Skript)
# ------------------------------
meta = {
    "var": var_type,
    "unit": enc["unit"],
    "dtype": "uint8",
    "nodata": NODATA_U8,
    "levels": LEVELS,
    "transform": enc["transform"],
    "vmin": enc["vmin"],
    "vmax": enc["vmax"],
    "decode": {
        "linear": "v = vmin + (q/254) * (vmax - vmin)",
        "sqrt": "v = vmin + (q/254)^2 * (vmax - vmin)",
        "ssqrt": "u = q/254*2 - 1; v = sign(u) * u^2 * vmax",
        "raw": "v = q",
    }[enc["transform"]],
    "orientation": "row0 = Norden, row-major, uint8",
    "projection": "EPSG:4326",
    "extent_4326": [GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX],
    "width": GRID_WIDTH,
    "height": GRID_HEIGHT,
    "files": [],
}


def read_crop(node):
    """Liest den Ausschnitt eines 2D-Kindes (lat, lon) als float32.
    Rueckgabe: (data_float32, lat_aufsteigend, lon)"""
    nlat, nlon = node.shape  # 2D, kein Zeitindex - ein Zeitschritt pro Datei
    row_start, row_end, col_start, col_end, lat_res, lon_res = compute_crop_indices(nlat, nlon)
    lat_crop, lon_crop = crop_lat_lon_arrays(row_start, row_end, col_start, col_end, lat_res, lon_res, nlat)

    sl = (slice(row_start, row_end + 1), slice(col_start, col_end + 1))
    data = np.asarray(node.read_array(sl), dtype=np.float32)

    if LAT_STORED_DESCENDING:
        data = data[::-1, :]   # Zeile 0 -> Sueden, wie RegularGridInterpolator erwartet
        lat_crop = lat_crop[::-1]
    return data, lat_crop, lon_crop


# ------------------------------
# Dateien durchgehen - JEDE .om Datei ist ein eigener Zeitschritt und
# enthaelt ALLE Variablen als Kinder.
# ------------------------------
def process_file(filename):
    """Verarbeitet EINE .om Datei. In eigene Funktion ausgelagert, damit alle
    lokalen Arrays nach dem Aufruf aus dem Scope fallen (stabiler Speicher)."""
    om_path = os.path.join(data_dir, filename)

    try:
        valid_time_utc = parse_valid_time_from_filename(filename)
    except ValueError as e:
        print(f"{e} - überspringe")
        return

    with OmFileReader(om_path) as root:
        if not root.is_group:
            print(f"{om_path}: root ist keine Gruppe (is_array={root.is_array}) - überspringe")
            return

        if var_type == "wwhd":
            try:
                precip_node = root.get_child_by_name("precipitation")
                snow_node = root.get_child_by_name("snowfall_water_equivalent")
            except Exception as e:
                print(f"{om_path}: Kind nicht gefunden - überspringe ({e})")
                return

            precip_raw, lat_asc, lon_crop = read_crop(precip_node)
            snow_raw, _, _ = read_crop(snow_node)

            data = calculate_ww(precip_raw, snow_raw, dt_seconds=DT_SECONDS)
            del precip_raw, snow_raw
        else:
            try:
                var_node = root.get_child_by_name(child_name)
            except Exception as e:
                print(f"{om_path}: Kind '{child_name}' nicht gefunden - überspringe ({e})")
                return

            if not var_node.is_array:
                print(f"{om_path}: '{child_name}' ist kein Array - überspringe")
                return

            data_raw, lat_asc, lon_crop = read_crop(var_node)
            data = convert(data_raw.astype(np.float64))
            del data_raw

    # ------------------------------
    # Quellgitter -> Ausgaberaster -> uint8
    # ------------------------------
    grid_data = to_target_grid(data, lon_crop, lat_asc, method=INTERP_METHOD)   # row0 = Norden
    del data
    q = encode_uint8(grid_data, enc)
    del grid_data

    valid_time_local = valid_time_utc.astimezone(ZoneInfo("Europe/Berlin"))
    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}"
    out_path = os.path.join(output_dir, outname)
    np.ascontiguousarray(q, dtype=np.uint8).tofile(out_path)

    meta["files"].append({
        "file": outname,
        "valid_time_utc": valid_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_time_local": valid_time_local.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    print(f"{filename} -> {outname} ({GRID_WIDTH}x{GRID_HEIGHT}, {q.nbytes} Bytes)")


all_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".om"))

if not all_files:
    print(f"Keine .om Dateien in {data_dir} gefunden")

for i, filename in enumerate(all_files):
    process_file(filename)
    # Alle 10 Dateien den Garbage Collector anstossen (viele Zeitschritte beim 15min-Modell).
    if i % 10 == 9:
        gc.collect()

print("META: " + json.dumps(meta, ensure_ascii=False))
print("Fertig!")