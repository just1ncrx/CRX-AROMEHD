import sys
import os
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
data_dir = sys.argv[1]        # z.B. "output"
output_dir = sys.argv[2]      # z.B. "output/maps"
var_type = sys.argv[3]        # 't2m', 'wind', ...
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# var_type -> Substring, der im Dateinamen der zugehoerigen .om Datei steht
# ------------------------------
OM_FILENAME_PATTERNS = {
    "t2m": "temperature_2m",
    "wind": "wind_gusts_10m",
    "tp": "precipitation",
    "cape_ml": "cape"
}

# ------------------------------
# Fuer welche Variablen die echten Werte zusaetzlich als DVAL-Chunk
# ins WebP eingebettet werden sollen (kein separates File noetig).
# ------------------------------
EMBED_DATA_VARS = {"t2m", "wind", "tp", "cape_ml"}

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

COLORMAPS = {
    "t2m": (t2m_colors, t2m_norm),
    "wind": (wind_colors, wind_norm),
    "tp": (prec_colors, prec_norm),
    "cape_ml": (cape_colors, cape_norm)
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

# Dimensionsreihenfolge im om-Array. Bestaetigt: shape=(721,1440,85), Zeit
# ist die LETZTE Dimension (siehe auch 'coordinates'-Kind: "lat lon time").
DIM_ORDER = ("lat", "lon", "time")

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


def load_valid_times(root, ntime, om_path):
    """Liest die echten (nicht-gleichmaessigen) Zeitschritte direkt aus dem
    'time'-Kind der om-Datei (Unix-Timestamps in Sekunden, UTC)."""
    try:
        time_child = root.get_child_by_name("time")
    except Exception as e:
        raise ValueError(
            f"{om_path}: kein 'time'-Kind gefunden - kann Zeitstempel nicht bestimmen ({e})"
        )

    if not time_child.is_array or time_child.shape[0] != ntime:
        raise ValueError(
            f"{om_path}: 'time'-Kind passt nicht (shape={time_child.shape}, erwartet ntime={ntime})"
        )

    raw = time_child.read_array((slice(0, ntime),))
    return [dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc) for t in raw]


# ------------------------------
# Farb-/Konvertierungs-Auswahl fuer den angeforderten var_type
# ------------------------------
if var_type not in COLORMAPS:
    print(f"Unbekannter var_type {var_type}")
    sys.exit(1)

cmap, norm = COLORMAPS[var_type]
convert = UNIT_CONVERT.get(var_type, lambda v: v)

pattern = OM_FILENAME_PATTERNS.get(var_type)
if pattern is None:
    print(f"var_type '{var_type}' hat noch kein Dateinamen-Muster in OM_FILENAME_PATTERNS")
    sys.exit(1)

# ------------------------------
# Dateien durchgehen
# ------------------------------
all_files_global = sorted(f for f in os.listdir(data_dir) if f.endswith(".om"))
matching_files = [f for f in all_files_global if pattern.lower() in f.lower()]

if not matching_files:
    print(f"Keine .om Datei in {data_dir} gefunden, die zu '{pattern}' passt "
          f"(gefunden: {all_files_global})")

for filename in matching_files:
    om_path = os.path.join(data_dir, filename)

    with OmFileReader(om_path) as root:
        # root ist gleichzeitig das Datenarray UND hat Metadaten-Kinder
        # (time, crs_wkt, unit, forecast_reference_time, coordinates, created_at).
        if not root.is_array:
            print(f"{om_path}: root ist kein Array (is_group={root.is_group}) - überspringe")
            continue

        nlat, nlon, ntime = root.shape  # bestaetigt: (lat, lon, time)

        row_start, row_end, col_start, col_end, lat_res, lon_res = compute_crop_indices(nlat, nlon)
        lat_crop, lon_crop = crop_lat_lon_arrays(row_start, row_end, col_start, col_end, lat_res, lon_res, nlat)

        # Ganzen Europa-Ausschnitt fuer ALLE Zeitschritte auf einmal lesen -
        # passt zum Chunk-Layout (Zeitachse wird ohnehin komplett pro Chunk
        # gespeichert), spart also viele einzelne read_array-Aufrufe.
        data_all = root.read_array((
            slice(row_start, row_end + 1),
            slice(col_start, col_end + 1),
            slice(0, ntime),
        ))  # shape: (nrows, ncols, ntime)

        valid_times_utc = load_valid_times(root, ntime, om_path)

        for t_idx in range(ntime):
            data = convert(np.asarray(data_all[:, :, t_idx], dtype=np.float64))

            if LAT_STORED_DESCENDING:
                data = data[::-1, :]  # Zeile 0 -> Sueden, wie warp_equirect erwartet
                lat_asc = lat_crop[::-1]
            else:
                lat_asc = lat_crop

            # ------------------------------
            # Nach EPSG:3857 (Web Mercator) umprojizieren
            # ------------------------------
            render_data_merc = warp_equirect_to_webmercator(data, lon_crop, lat_asc, extent, method="linear")

            # ------------------------------
            # Transparentes WebP speichern
            # ------------------------------
            outname = f"{var_type}_{valid_times_utc[t_idx].astimezone(ZoneInfo('Europe/Berlin')):%Y%m%d_%H%M}.webp"
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

            print(f"{filename} t_idx={t_idx} -> {outname}")
