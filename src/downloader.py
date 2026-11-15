"""Download hourly ERA5 weather data from the Open-Meteo Historical Weather API.

Design:
* one HTTP request = one city x one decade (keeps responses ~6 MB and lets a
  failed request be retried cheaply);
* each chunk is written to data/raw/<city>/<city>_<start>_<end>.csv, so the
  download can be interrupted and resumed (already present chunks are skipped);
* HTTP 429 (rate limit) and network errors are retried with exponential backoff;
* progress is saved to data/download_state.json (program state on disk).
"""
import csv
import time
from datetime import date

import requests

import config
from src.utils import load_json, save_json


def decade_chunks(start: date, end: date, years: int = config.CHUNK_YEARS) -> list[tuple[date, date]]:
    """Split [start, end] into consecutive periods aligned to decades."""
    chunks = []
    cur = start
    while cur <= end:
        nxt_year = (cur.year // years + 1) * years          # next decade boundary
        chunk_end = min(date(nxt_year - 1, 12, 31), end)
        chunks.append((cur, chunk_end))
        cur = date(nxt_year, 1, 1)
    return chunks


def build_params(city: str, start: date, end: date) -> dict:
    """Query parameters for one API request."""
    meta = config.CITIES[city]
    return {
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(config.HOURLY_VARIABLES),
        "models": config.API_MODEL,
        "timezone": "GMT",
    }


class DailyLimitError(RuntimeError):
    """The free daily quota (10 000 weighted calls) is used up - resume tomorrow."""


def fetch_with_retry(params: dict, session: requests.Session) -> dict:
    """GET the API, retrying on rate limits (429), 5xx and connection errors.

    Minutely/hourly limits are waited out. The daily limit is retried a few
    times (behind a shared NAT another exit IP may still have quota) and then
    raises DailyLimitError, so the run stops cleanly and can be resumed.
    """
    delay = 5
    daily_hits = 0
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = session.get(config.API_URL, params=params, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                reason = resp.json().get("reason", "") if resp.headers.get("content-type", "").startswith("application/json") else ""
                if "Daily" in reason:
                    daily_hits += 1
                    if daily_hits >= 4:
                        raise DailyLimitError(reason)
                if "Hourly" in reason:
                    delay = max(delay, 300)
                print(f"    HTTP {resp.status_code} {reason[:60]} -> retry {attempt}/{config.MAX_RETRIES} in {delay}s")
            else:
                # 4xx other than 429 means a bad request: retrying will not help
                raise ValueError(f"API error {resp.status_code}: {resp.text[:200]}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            print(f"    network error {type(exc).__name__} -> retry {attempt}/{config.MAX_RETRIES} in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up after {config.MAX_RETRIES} attempts: {params['start_date']}..{params['end_date']}")


def save_chunk_csv(payload: dict, city: str, path) -> int:
    """Write one API response to CSV (city, time + hourly variables). Returns rows."""
    hourly = payload["hourly"]
    cols = ["time"] + config.HOURLY_VARIABLES
    n = len(hourly["time"])
    tmp = path.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["city"] + cols)
        for i in range(n):
            writer.writerow([city] + [hourly[c][i] if hourly[c][i] is not None else "" for c in cols])
    tmp.replace(path)          # atomic rename: no half-written CSV is ever left behind
    return n


def download_city(city: str, session: requests.Session, state: dict) -> int:
    """Download all decades for one city; returns number of new rows."""
    city_dir = config.RAW_DIR / city
    city_dir.mkdir(parents=True, exist_ok=True)
    new_rows = 0
    for start, end in decade_chunks(config.START_DATE, config.END_DATE):
        path = city_dir / f"{city}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        if path.exists() and path.stat().st_size > 0:
            continue                                   # resume support
        print(f"  {city}: {start} .. {end}")
        payload = fetch_with_retry(build_params(city, start, end), session)
        rows = save_chunk_csv(payload, city, path)
        new_rows += rows
        state.setdefault("cities", {}).setdefault(city, {})
        state["cities"][city].update({
            "grid_lat": payload.get("latitude"),
            "grid_lon": payload.get("longitude"),
            "elevation": payload.get("elevation"),
        })
        state["cities"][city].setdefault("chunks", {})[path.name] = rows
        save_json(state, config.STATE_FILE)
        # be polite: one chunk weighs ~100 calls, the minutely limit is 600
        time.sleep(6.0)
    return new_rows


def download_all(cities: list[str] | None = None) -> dict:
    """Download every configured city (or a subset). Returns the state dict."""
    cities = cities or list(config.CITIES)
    state = load_json(config.STATE_FILE, default={}) or {}
    state["source"] = config.API_URL
    state["model"] = config.API_MODEL
    state["period"] = [config.START_DATE.isoformat(), config.END_DATE.isoformat()]
    total = 0
    with requests.Session() as session:
        for city in cities:
            if city not in config.CITIES:
                print(f"  ! unknown city {city!r}, skipped")
                continue
            try:
                total += download_city(city, session, state)
            except DailyLimitError:
                print("  ! Open-Meteo daily limit reached. Already downloaded chunks are kept;\n"
                      "    run the download again tomorrow - it resumes where it stopped.")
                break
            except (RuntimeError, ValueError) as exc:
                print(f"  ! {city} failed: {exc}. Re-run later to resume.")
    print(f"Downloaded {total:,} new hourly rows")
    return state


if __name__ == "__main__":
    download_all()
