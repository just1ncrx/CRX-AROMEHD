# Gemeinsame Kodierung: wird von generate_pngs.py UND generate_metadata.py benutzt.
# Stufen: 0..254 = Wert, 255 = NoData.
# transform:
#   "linear": u = (v - vmin) / (vmax - vmin)
#   "sqrt"  : u = sqrt((v - vmin) / (vmax - vmin))
#   "ssqrt" : u = (sign(v)*sqrt(|v|/vmax) + 1) / 2   (vmin = -vmax)
#   "raw"   : q = Wert direkt (nur für Wettercodes)
NODATA_U8 = 255
LEVELS = 254

ENCODING = {
    "t2m":     {"vmin": -35.0, "vmax": 45.0,   "transform": "linear", "unit": "°C"},
    "wind":    {"vmin": 0.0,   "vmax": 250.0,  "transform": "linear", "unit": "km/h"},
    "tp15":    {"vmin": 0.0,   "vmax": 125.0,  "transform": "sqrt",   "unit": "mm"},
    "cape_ml": {"vmin": 0.0,   "vmax": 3000.0, "transform": "sqrt",   "unit": "J/kg"},
    "wwhd":    {"vmin": 0.0,   "vmax": 254.0,  "transform": "raw",    "unit": "code"},
}