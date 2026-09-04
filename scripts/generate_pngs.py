import sys
import os
import re
import gc
import struct
import zlib
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from PIL import Image
from omfiles import OmFileReader

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]        # z.B. "data/parameter"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3]        # 't2m', 'wind', 'ww', ...
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Zeitschrittlaenge des Modells in Sekunden. Anpassen falls sich das
# je nach Modell/Layout unterscheidet (z.B. 900 fuer 15min-Modell).
# ------------------------------
DT_SECONDS = 900

# ------------------------------
# var_type -> Name des Kindes (Variable) INNERHALB jeder .om Datei.
# Im data_spatial-Layout enthaelt JEDE Datei ALLE Variablen fuer genau
# einen Zeitschritt (root ist eine Gruppe, kein Array). Der Dateiname
# selbst ist der Zeitstempel, z.B. "2026-09-03T1900.om".
# "ww" wird separat behandelt (braucht zwei Kinder), steht deshalb NICHT
# hier drin.
# ------------------------------
OM_CHILD_NAMES = {
    "t2m": "temperature_2m",
    "wind": "wind_gusts_10m",
    "tp": "precipitation",
    "cape_ml": "cape",
}

# Regex fuer den Zeitstempel im Dateinamen: YYYY-MM-DDTHHMM.om
FILENAME_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{4})")

# ------------------------------
# Fuer welche Variablen die echten Werte zusaetzlich als DVAL-Chunk
# ins WebP eingebettet werden sollen (kein separates File noetig).
# ------------------------------
EMBED_DATA_VARS = {"t2m", "wind"}

# ------------------------------
# Temperatur-Farben
# ------------------------------
t2m_bounds = list(range(-36, 50, 2))
t2m_colors = LinearSegmentedColormap.from_list(
    "t2m_smoooth",
    [
        "#F675F4", "#F428E9", "#B117B5", "#950CA2", "#640180",
        "#3E007F", "#00337E", "#005295", "#1292FF", "#49ACFF",
        "#8FCDFF", "#B4DBFF", "#B9ECDD", "#88D4AD", "#07A125",
        "#3FC107", "#9DE004", "#E7F700", "#F3CD0A", "#EE5505",
        "#C81904", "#AF0E14", "#620001", "#C87879", "#FACACA",
        "#E1E1E1", "#6D6D6D"
    ],
    N=len(t2m_bounds)
)
t2m_norm = BoundaryNorm(t2m_bounds, ncolors=len(t2m_bounds))

# ------------------------------
# Windböen-Farben
# ------------------------------
wind_bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300]
wind_colors = ListedColormap([
    "#68AD05", "#8DC00B", "#B1D415", "#D5E81C", "#FBFC22",
    "#FAD024", "#F9A427", "#FC7929", "#FB4D2B", "#EA2B57",
    "#FB22A5", "#FC22CE", "#FC22F5", "#FC62F8", "#FD80F8",
    "#FFBFFC", "#FEDFFE", "#FEFFFF", "#E1E0FF", "#C3C3FF",
    "#A5A5FF", "#A5A5FF", "#6868FE"
])
wind_norm = mcolors.BoundaryNorm(wind_bounds, wind_colors.N)

# ------------------------------
# Niederschlags-Farben 1h (tp)
# ------------------------------
prec_bounds = [0.0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#FFFFFF", "#B4D7FF", "#75BAFF", "#349AFF", "#0582FF", "#0069D2",
    "#003680", "#148F1B", "#1ACF06", "#64ED07", "#FFF32B",
    "#E9DC01", "#F06000", "#FF7F26", "#FFA66A", "#F94E78",
    "#F71E53", "#BE0000", "#880000", "#64007F", "#C201FC",
    "#DD66FE", "#EBA6FF", "#F9E7FF", "#D4D4D4"
])
prec_norm = mcolors.BoundaryNorm(prec_bounds, prec_colors.N)

# ------------------------------
# CAPE-Farben
# ------------------------------
cape_bounds = [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
cape_colors = ListedColormap([
    "#676767", "#006400", "#008000", "#00CC00", "#66FF00", "#FFFF00",
    "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000", "#FF0095",
    "#FC439F", "#FF88D3", "#FF99FF"
])
cape_norm = mcolors.BoundaryNorm(cape_bounds, cape_colors.N)

# ------------------------------
# ww-Farben (nur Regen/Schneeregen/Schnee, Rest = 0 = grau)
# ------------------------------
ww_colors_base = {
    0:  "#696969",   # alles Sonstige / kein Niederschlag
    51: "#C2FF9A",   # Nieselregen
    56: "#FFA500", 57: "#C06A00",   # Schneeregen
    61: "#00FF00", 63: "#00C300", 65: "#009700",   # Regen
    71: "#ADD8E6", 73: "#6495ED", 75: "#00008B",   # Schnee
}

# var_type -> (cmap, norm). "ww" ist absichtlich NICHT hier drin, weil es
# den eigenen Lookup-Pfad ww_to_rgba() statt cmap/norm nutzt.
COLORMAPS = {
    "t2m": (t2m_colors, t2m_norm),
    "wind": (wind_colors, wind_norm),
    "tp": (prec_colors, prec_norm),
    "cape_ml": (cape_colors, cape_norm),
}

# Umrechnung Rohwert -> Anzeige-Einheit, je Variable
UNIT_CONVERT = {
    "t2m": lambda v: v - 273.15 if np.nanmax(v) > 200 else v,  # K -> °C, nur falls noetig
    "wind": lambda v: v * 3.6,  # m/s -> km/h
}

# Quantisierungsschritt je Variable fuer den eingebetteten DVAL-Chunk
# (feiner als die Anzeige-Nachkommastellen, damit kein sichtbarer
# Genauigkeitsverlust entsteht).
QUANTUM_STEP = {
    "t2m": 0.05,   # °C
    "wind": 0.2,   # km/h
}
NAN_SENTINEL_I16 = -32768
DVAL_FOURCC = b"DVAL"

# ------------------------------
# Globale Bbox der Quelldateien (bestaetigt via crs_wkt-Kind der .om Datei)
# ------------------------------
GLOBAL_LAT_MIN, GLOBAL_LON_MIN = 37.5, -12.0
GLOBAL_LAT_MAX, GLOBAL_LON_MAX = 55.4, 16.0

# Speicherreihenfolge der Lat-Achse in der Datei (Zeile 0 = Norden oder Sueden?).
# Per check_lat_order.py verifiziert: Zeile 0 = ~-52°C (Suedpol) -> aufsteigend gespeichert.
LAT_STORED_DESCENDING = False

# Dimensionsreihenfolge der Variablen-Arrays im data_spatial-Layout:
# JEDE Datei ist EIN Zeitschritt, jede Variable ist ein 2D-Array (lat, lon)
# OHNE eigene Zeitachse.
DIM_ORDER = ("lat", "lon")

# ------------------------------
# Bounding Box (wie im GRIB2-Skript)
# ------------------------------
extent = [-3.94, 20.34, 43.18, 58.08]  # lon_min, lon_max, lat_min, lat_max

# Bounding Box fuer den eingebetteten DVAL-Chunk (nur t2m/wind) - hier
# reicht Deutschland + etwas Rand fuer Grenzregionen beim Hovern; das
# Farbbild selbst bleibt unveraendert auf der vollen Domaene.
GERMANY_BBOX_LONLAT = [5.5, 15.3, 47.0, 55.3]  # lon_min, lon_max, lat_min, lat_max

# ------------------------------
# EPSG:4326 -> EPSG:3857 (Web Mercator)
# ------------------------------
EARTH_RADIUS = 6378137.0
WEBMERCATOR_WIDTH = 1024


def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH):
    lon_min, lon_max, lat_min, lat_max = extent
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_height = max(int(round(out_width * aspect)), 1)
    x_new = np.linspace(x_min, x_max, out_width)
    y_new = np.linspace(y_min, y_max, out_height)  # aufsteigend: Süd -> Nord
    return x_new, y_new


def warp_equirect_to_webmercator(data, lon, lat, extent, method="linear", out_width=WEBMERCATOR_WIDTH):
    """data/lon/lat: reguläres lat/lon-Gitter, lat und lon aufsteigend sortiert
    (Zeile 0 = Süden)."""
    x_new, y_new = webmercator_target_grid(extent, out_width=out_width)
    xx, yy = np.meshgrid(x_new, y_new)
    lon_grid = np.degrees(xx / EARTH_RADIUS)
    lat_grid = np.degrees(2 * np.arctan(np.exp(yy / EARTH_RADIUS)) - np.pi / 2)

    interp_func = RegularGridInterpolator(
        (lat, lon), data, method=method, bounds_error=False, fill_value=np.nan
    )
    pts = np.array([lat_grid.ravel(), lon_grid.ravel()]).T
    return interp_func(pts).reshape(lat_grid.shape)


# Gleiches Ziel-Pixelraster wie in warp_equirect_to_webmercator (muss mit
# WEBMERCATOR_WIDTH uebereinstimmen, damit die Indizes exakt passen) -
# einmalig ausserhalb der Schleife berechnet, da pro Lauf identisch.
_full_x_new, _full_y_new = webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH)

_gbx_min, _gby_min = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[0], GERMANY_BBOX_LONLAT[2])
_gbx_max, _gby_max = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[1], GERMANY_BBOX_LONLAT[3])

# Indizes im vollen Raster, die die Bbox gerade so umschliessen (lieber
# ein Pixel zu viel als zu wenig - daher aussen aufrunden statt clippen).
_col_i0 = max(0, np.searchsorted(_full_x_new, _gbx_min, side="left") - 1)
_col_i1 = min(len(_full_x_new) - 1, np.searchsorted(_full_x_new, _gbx_max, side="right"))
_row_i0 = max(0, np.searchsorted(_full_y_new, _gby_min, side="left") - 1)
_row_i1 = min(len(_full_y_new) - 1, np.searchsorted(_full_y_new, _gby_max, side="right"))

# Exakte Mercator-Extent des zugeschnittenen Rasters (= tatsaechliche
# Gitterpunkte an den Raendern, nicht die rohe Bbox - damit die
# Ruecktransformation im Frontend pixelgenau bleibt).
GERMANY_CROP_EXTENT_3857 = [
    float(_full_x_new[_col_i0]), float(_full_y_new[_row_i0]),
    float(_full_x_new[_col_i1]), float(_full_y_new[_row_i1]),
]


def crop_to_germany(data_south_first):
    """data_south_first: 2D-Array wie von warp_equirect_to_webmercator
    zurückgegeben (row0 = Süden, aufsteigend in Mercator-Y wie
    _full_y_new). Schneidet auf die Deutschland-Bbox zu."""
    return data_south_first[_row_i0:_row_i1 + 1, _col_i0:_col_i1 + 1]


def data_to_rgba(data, cmap, norm):
    rgba = cmap(norm(data))
    rgba = (rgba * 255).astype(np.uint8)
    rgba[~np.isfinite(data), 3] = 0
    return rgba


def save_transparent_webp(data, cmap, norm, out_path):
    rgba = data_to_rgba(data, cmap, norm)
    img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")  # Zeile 0 -> oben = Norden
    img.save(out_path, format="WEBP", lossless=True, method=4)


def embed_data_chunk(webp_path, data, extent_3857, quantum, fourcc=DVAL_FOURCC):
    """Hängt ein rohes Datenfeld als privaten, int16-quantisierten RIFF-Chunk
    an ein WebP an.

    data: 2D-Array (float), row0 = Norden (also bereits wie fürs Bild
          gespiegelt).
    extent_3857: [x_min, y_min, x_max, y_max] in Web-Mercator-Metern -
                 exakt das Raster, auf dem `data` liegt.
    quantum: Rasterschritt in den Originaleinheiten (z.B. 0.05 für °C).
    """
    height, width = data.shape

    nan_mask = ~np.isfinite(data)
    data_filled = np.where(nan_mask, 0.0, data)  # verhindert NaN->int Warnung beim Runden/Casten
    quant = np.round(data_filled / quantum)
    # Sicherheitsclip: verhindert einen int16-Überlauf bei extremen
    # Ausreißern, ohne das eigentlich zulässige Wertespektrum
    # (t2m/wind liegen weit darunter) einzuschränken.
    quant = np.clip(quant, -32767, 32767).astype(np.int16)
    quant[nan_mask] = NAN_SENTINEL_I16

    header = struct.pack("<BBII", 2, 1, width, height)
    header += struct.pack("<4d", *extent_3857)
    header += struct.pack("<d", quantum)
    compressed = zlib.compress(np.ascontiguousarray(quant, dtype="<i2").tobytes(), level=9)
    payload = header + compressed

    size = len(payload)
    chunk = fourcc + struct.pack("<I", size) + payload
    if size % 2 == 1:
        chunk += b"\x00"  # RIFF-Padding auf gerade Länge, zählt nicht zu size

    with open(webp_path, "rb") as f:
        content = f.read()

    if content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
        raise ValueError(f"{webp_path} ist keine gültige WebP-Datei (RIFF/WEBP-Header fehlt)")

    riff_size = struct.unpack("<I", content[4:8])[0]
    new_riff_size = riff_size + len(chunk)

    with open(webp_path, "wb") as f:
        f.write(content[:4])
        f.write(struct.pack("<I", new_riff_size))
        f.write(content[8:])
        f.write(chunk)


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

    Rückgabe: float-Array mit Codes 0, 56/57, 61/63/65, 71/73/75, oder
    NaN dort, wo precip/snow_we selbst NaN war (= außerhalb des
    AROME-Modellgebiets, nicht "kein Niederschlag"!).
    """
    dt_h = dt_seconds / 900.0

    # Wo mindestens einer der beiden Eingabewerte fehlt (außerhalb des
    # Modellgebiets) -> merken und am Ende NaN reinschreiben.
    invalid = ~np.isfinite(precip) | ~np.isfinite(snow_we)

    # Fuer die Vergleiche unten duerfen keine NaN mehr durch, sonst wuerden
    # alle Bedingungen False bleiben und der Default-Code 0 (= "kein
    # Niederschlag", opak grau) faelschlich stehen bleiben. Wird gleich
    # durch die invalid-Maske wieder ueberschrieben.
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

    # Ganz zum Schluss: außerhalb des Modellgebiets -> NaN statt Code 0,
    # damit ww_to_rgba() das korrekt transparent zeichnet.
    code[invalid] = np.nan

    return code


def ww_to_rgba(code_array, colors_hex):
    lut_max = max(colors_hex) + 1
    table = np.zeros((lut_max, 4), dtype=np.uint8)
    for k, v in colors_hex.items():
        table[k] = tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) + (255,)

    valid = np.isfinite(code_array)
    idx = np.clip(np.where(valid, code_array, 0).astype(np.int32), 0, lut_max - 1)
    rgba = table[idx]
    rgba[~valid, 3] = 0
    return rgba


# ------------------------------
# Farb-/Konvertierungs-Auswahl fuer den angeforderten var_type
# "ww" ist ein Sonderfall: kein Eintrag in COLORMAPS/OM_CHILD_NAMES,
# weil es zwei Kinder liest und einen eigenen Rendering-Pfad hat.
# ------------------------------
if var_type != "wwhd" and var_type not in COLORMAPS:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)

if var_type != "wwhd":
    cmap, norm = COLORMAPS[var_type]
    convert = UNIT_CONVERT.get(var_type, lambda v: v)

    child_name = OM_CHILD_NAMES.get(var_type)
    if child_name is None:
        print(f"var_type '{var_type}' hat noch kein Kind-Name-Mapping in OM_CHILD_NAMES")
        sys.exit(1)


# ------------------------------
# Dateien durchgehen - JEDE .om Datei ist ein eigener Zeitschritt und
# enthaelt ALLE Variablen als Kinder, daher KEIN Filtern nach
# Variablenname im Dateinamen mehr (wie es beim alten data_run-Layout
# noetig war).
# ------------------------------
def process_file(filename):
    """Verarbeitet EINE .om Datei. In eine eigene Funktion ausgelagert,
    damit alle lokalen Arrays (data, render_data_merc, ...) garantiert
    aus dem Scope fallen, sobald der Aufruf zurueckkehrt - das haelt den
    Speicherverbrauch ueber viele Dateien hinweg stabil, statt dass sich
    Referenzen im globalen Namespace der for-Schleife ansammeln."""
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

        # ------------------------------
        # Sonderfall ww: braucht zwei Kinder, eigener Rendering-Pfad.
        # Alles innerhalb des "with"-Blocks, damit root/get_child_by_name
        # noch gueltig ist.
        # ------------------------------
        if var_type == "wwhd":
            try:
                precip_node = root.get_child_by_name("precipitation")
                snow_node = root.get_child_by_name("snowfall_water_equivalent")
            except Exception as e:
                print(f"{om_path}: Kind nicht gefunden - überspringe ({e})")
                return

            nlat, nlon = precip_node.shape

            row_start, row_end, col_start, col_end, lat_res, lon_res = compute_crop_indices(nlat, nlon)
            lat_crop, lon_crop = crop_lat_lon_arrays(row_start, row_end, col_start, col_end, lat_res, lon_res, nlat)

            sl = (slice(row_start, row_end + 1), slice(col_start, col_end + 1))
            precip_raw = np.asarray(precip_node.read_array(sl), dtype=np.float32)
            snow_raw = np.asarray(snow_node.read_array(sl), dtype=np.float32)

            data = calculate_ww(precip_raw, snow_raw, dt_seconds=DT_SECONDS)
            del precip_raw, snow_raw

            if LAT_STORED_DESCENDING:
                data = data[::-1, :]
                lat_asc = lat_crop[::-1]
            else:
                lat_asc = lat_crop

            render_data_merc = warp_equirect_to_webmercator(
                data.astype(np.float32), lon_crop, lat_asc, extent, method="nearest"  # WICHTIG: nearest!
            )
            del data

            outname = f"{var_type}_{valid_time_utc.astimezone(ZoneInfo('Europe/Berlin')):%Y%m%d_%H%M}.webp"
            out_path = os.path.join(output_dir, outname)
            rgba = ww_to_rgba(render_data_merc, ww_colors_base)
            img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")
            img.save(out_path, format="WEBP", lossless=True, method=4)
            print(f"{filename} -> {outname}")
            return  # fertig, Rest der Funktion (generischer Pfad) ueberspringen

        # ------------------------------
        # Generischer Pfad fuer t2m/wind/tp/cape_ml
        # ------------------------------
        try:
            var_node = root.get_child_by_name(child_name)
        except Exception as e:
            print(f"{om_path}: Kind '{child_name}' nicht gefunden - überspringe ({e})")
            return

        if not var_node.is_array:
            print(f"{om_path}: '{child_name}' ist kein Array - überspringe")
            return

        nlat, nlon = var_node.shape  # 2D, kein Zeitindex mehr - ein Zeitschritt pro Datei

        row_start, row_end, col_start, col_end, lat_res, lon_res = compute_crop_indices(nlat, nlon)
        lat_crop, lon_crop = crop_lat_lon_arrays(row_start, row_end, col_start, col_end, lat_res, lon_res, nlat)

        data_raw = var_node.read_array((
            slice(row_start, row_end + 1),
            slice(col_start, col_end + 1),
        ))  # shape: (nrows, ncols)

        # float32 statt float64 halbiert den Speicherbedarf fuer diese
        # (in der Schleife immer wieder neu angelegten) Arrays.
        data = convert(np.asarray(data_raw, dtype=np.float32))
        del data_raw

        if LAT_STORED_DESCENDING:
            data = data[::-1, :]  # Zeile 0 -> Sueden, wie warp_equirect erwartet
            lat_asc = lat_crop[::-1]
        else:
            lat_asc = lat_crop

        # ------------------------------
        # Nach EPSG:3857 (Web Mercator) umprojizieren
        # ------------------------------
        render_data_merc = warp_equirect_to_webmercator(data, lon_crop, lat_asc, extent, method="linear")
        del data

        # ------------------------------
        # Transparentes WebP speichern
        # ------------------------------
        outname = f"{var_type}_{valid_time_utc.astimezone(ZoneInfo('Europe/Berlin')):%Y%m%d_%H%M}.webp"
        out_path = os.path.join(output_dir, outname)
        save_transparent_webp(render_data_merc, cmap, norm, out_path)

        # Fuer t2m/wind zusaetzlich die echten physikalischen Werte
        # (°C bzw. km/h, nicht die Farben) als privaten RIFF-Chunk
        # direkt ins WebP einbetten - row0 = Norden, damit der Chunk
        # 1:1 zur Bildorientierung passt (das Bild wird in
        # save_transparent_webp beim Speichern gespiegelt,
        # render_data_merc selbst hat row0 = Sueden).
        if var_type in EMBED_DATA_VARS:
            germany_data = crop_to_germany(render_data_merc)          # row0 = Süden
            quantum = QUANTUM_STEP.get(var_type, 0.1)
            embed_data_chunk(out_path, germany_data[::-1], GERMANY_CROP_EXTENT_3857, quantum)  # row0 = Norden

        print(f"{filename} -> {outname}")


all_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".om"))

if not all_files:
    print(f"Keine .om Dateien in {data_dir} gefunden")

for i, filename in enumerate(all_files):
    process_file(filename)
    # Alle 10 Dateien den Garbage Collector explizit anstossen, damit sich
    # ueber viele Iterationen kein Speicher unnoetig anstaut (relevant bei
    # sehr vielen Zeitschritten pro Lauf, z.B. beim 15min-Modell).
    if i % 10 == 9:
        gc.collect()
