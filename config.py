"""Project-wide configuration: cities, variables, paths and analysis constants.

Every module imports its settings from here, so changing a city list or a
date range is done in one place.
"""
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # raw CSV per city and decade (API output)
PROCESSED_DIR = DATA_DIR / "processed"     # cleaned Parquet files
HOURLY_PARQUET_DIR = PROCESSED_DIR / "hourly"   # partitioned by city
DAILY_PARQUET = PROCESSED_DIR / "daily.parquet"
DB_PATH = DATA_DIR / "kz_climate.db"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"
MODEL_DIR = ROOT / "models"
STATE_FILE = DATA_DIR / "download_state.json"

for _p in (RAW_DIR, PROCESSED_DIR, HOURLY_PARQUET_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- API
API_URL = "https://archive-api.open-meteo.com/v1/archive"
# ERA5 only (0.25°, 1940-present). The default "best_match" mixes ERA5,
# ERA5-Land and the IFS model after 2017, which would create an artificial
# jump in a long-term trend, so for climate work one consistent model is used.
API_MODEL = "era5"
START_DATE = date(1940, 1, 1)
# ERA5 is published with a ~5 day delay; keep a safety margin of 8 days.
END_DATE = date.today() - timedelta(days=8)
CHUNK_YEARS = 10                 # one request = one city x one decade
REQUEST_TIMEOUT = 120            # seconds
MAX_RETRIES = 12

HOURLY_VARIABLES = [
    "temperature_2m",            # °C
    "relative_humidity_2m",      # %
    "surface_pressure",          # hPa
    "precipitation",             # mm
]
# Why only 4 variables: the free API "weight" of a request is
#   nLocations * nDays/14 * nVariables/10,  limit 10 000 per day.
# 13 cities x 87 years x 4 variables ~ 11 800 calls (two days of quota);
# with 10 variables it would be ~29 500 calls. Dew point is derived from
# temperature and humidity (Magnus formula) instead of being downloaded.

# ---------------------------------------------------------------- cities
# name -> metadata. macro_zone groups cities for regional comparison,
# region is the administrative unit (oblast or city of republican status).
CITIES = {
    "Astana":    {"lat": 51.1694, "lon": 71.4491, "region": "Astana (city)",              "macro_zone": "North"},
    "Almaty":    {"lat": 43.2389, "lon": 76.8897, "region": "Almaty (city)",              "macro_zone": "South"},
    "Shymkent":  {"lat": 42.3417, "lon": 69.5901, "region": "Shymkent (city)",            "macro_zone": "South"},
    "Karaganda": {"lat": 49.8047, "lon": 73.1094, "region": "Karaganda Region",           "macro_zone": "Central"},
    "Aktobe":    {"lat": 50.2839, "lon": 57.1670, "region": "Aktobe Region",              "macro_zone": "West"},
    "Pavlodar":  {"lat": 52.2873, "lon": 76.9674, "region": "Pavlodar Region",            "macro_zone": "North"},
    "Oskemen":   {"lat": 49.9483, "lon": 82.6279, "region": "East Kazakhstan Region",     "macro_zone": "East"},
    "Atyrau":    {"lat": 47.1164, "lon": 51.8833, "region": "Atyrau Region",              "macro_zone": "West"},
    "Aktau":     {"lat": 43.6532, "lon": 51.1975, "region": "Mangystau Region",           "macro_zone": "West"},
    "Kostanay":  {"lat": 53.2144, "lon": 63.6246, "region": "Kostanay Region",            "macro_zone": "North"},
    "Kyzylorda": {"lat": 44.8488, "lon": 65.4823, "region": "Kyzylorda Region",           "macro_zone": "South"},
    "Petropavl": {"lat": 54.8657, "lon": 69.1469, "region": "North Kazakhstan Region",    "macro_zone": "North"},
    "Oral":      {"lat": 51.2333, "lon": 51.3667, "region": "West Kazakhstan Region",     "macro_zone": "West"},
}

# Kazakhstan has used a single time zone UTC+5 since March 2024. Data are
# downloaded in UTC and shifted by this fixed offset to build local days,
# which avoids historical DST jumps.
LOCAL_UTC_OFFSET_HOURS = 5

# ---------------------------------------------------------------- analysis
BASELINE_PERIOD = (1961, 1990)      # WMO reference period for anomalies/percentiles
RECENT_PERIOD = (1991, 2020)
WAVE_MIN_DAYS = 3                   # heat/cold wave = >= 3 consecutive days
HEAT_PCTL = 0.90                    # Tmax above calendar-day 90th percentile
COLD_PCTL = 0.10                    # Tmin below calendar-day 10th percentile
PCTL_WINDOW_DAYS = 15               # centred window for calendar-day percentiles

# forecasting split (chronological, no shuffling)
TRAIN_END = "2015-12-31"
VALID_END = "2020-12-31"            # 2016-2020 validation, 2021+ test
FORECAST_START = "1950-01-01"       # skip the first decade (sparser ERA5 input data)
RANDOM_STATE = 42
