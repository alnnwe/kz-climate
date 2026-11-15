"""Climate and Weather of Kazakhstan Cities - console application.

Run `python main.py` for the interactive menu or `python main.py --all`
to execute the whole pipeline without questions.
"""
import argparse
import multiprocessing as mp

import pandas as pd

import config

MENU = {
    "1": "Download hourly data from Open-Meteo (resumable)",
    "2": "Preprocess: CSV -> Parquet, dtype optimisation, quality audit, daily table",
    "3": "Climate analysis + figures (trends, extremes, heat/cold waves, STL, correlations)",
    "4": "Forecasting: baselines vs ARIMA vs Ridge vs gradient boosting",
    "5": "Load everything into SQLite",
    "6": "Run the 12 analytical SQL queries",
    "7": "Benchmarks: CSV vs Parquet, pandas vs SQL",
    "8": "City climate card: look up one city and year",
    "9": "Forecast tomorrow's temperature for every city",
    "10": "Run the full pipeline (steps 2-7)",
    "0": "Exit",
}


def ask_choice(prompt: str, options: list[str]) -> str:
    """Repeat the question until the user types one of the allowed options."""
    while True:
        value = input(prompt).strip()
        if value in options:
            return value
        print(f"  Please enter one of: {', '.join(options)}")


def ask_int(prompt: str, low: int, high: int) -> int:
    """Read an integer in [low, high]; invalid input is caught and asked again."""
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print(f"  '{raw}' is not a whole number, try again")
            continue
        if low <= value <= high:
            return value
        print(f"  Enter a value between {low} and {high}")


def ask_city() -> str:
    """Case-insensitive city choice by name or number."""
    names = list(config.CITIES)
    for i, n in enumerate(names, 1):
        print(f"  {i:2d}. {n}")
    while True:
        raw = input("City (name or number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        match = [n for n in names if n.lower() == raw.lower()]
        if match:
            return match[0]
        print("  Unknown city, try again")


def city_card() -> None:
    """Menu 8: climate summary of one city for one year compared with the 1961-1990 norm."""
    from src.analysis import add_calendar
    from src.preprocessing import load_daily

    city = ask_city()
    daily = add_calendar(load_daily([city]))
    year = ask_int(f"Year ({daily['year'].min()}-{daily['year'].max()}): ",
                   int(daily["year"].min()), int(daily["year"].max()))
    y = daily[daily["year"] == year]
    base = daily[daily["year"].between(*config.BASELINE_PERIOD)]
    card = {
        "Mean temperature, °C": (y["tmean"].mean(), base.groupby("year")["tmean"].mean().mean()),
        "Absolute max, °C": (y["tmax"].max(), base.groupby("year")["tmax"].max().mean()),
        "Absolute min, °C": (y["tmin"].min(), base.groupby("year")["tmin"].min().mean()),
        "Precipitation, mm": (y["precipitation"].sum(), base.groupby("year")["precipitation"].sum().mean()),
        "Frost days (Tmin<0)": ((y["tmin"] < 0).sum(), base.groupby("year")["tmin"].apply(lambda s: (s < 0).sum()).mean()),
        "Hot days (Tmax>=30)": ((y["tmax"] >= 30).sum(), base.groupby("year")["tmax"].apply(lambda s: (s >= 30).sum()).mean()),
    }
    print(f"\n  {city}, {year}" + (" (incomplete year)" if len(y) < 365 else ""))
    print(f"  {'indicator':<24}{'value':>10}{'1961-1990 norm':>16}{'difference':>12}")
    for k, (v, n) in card.items():
        print(f"  {k:<24}{v:>10.1f}{n:>16.1f}{v - n:>+12.1f}")
    monthly = y.groupby("month")["tmean"].mean()
    print("  monthly mean temperature, °C:",
          ", ".join(f"{m:02d}: {v:+.1f}" for m, v in monthly.items()))


def predict_tomorrow() -> None:
    """Menu 9: next-day forecast for every city with the saved model."""
    from src.forecasting import predict_next_day
    from src.preprocessing import load_daily

    out = predict_next_day(load_daily())
    print(out.to_string(index=False))
    path = config.TABLE_DIR / "next_day_forecast.csv"
    out.to_csv(path, index=False)
    print(f"  saved to {path}")


def menu() -> None:
    """Interactive console loop."""
    from src import pipeline

    actions = {
        "1": pipeline.step_download,
        "2": pipeline.step_preprocess,
        "3": pipeline.step_analysis,
        "4": lambda: pipeline.step_forecast(tune=ask_choice("Tune boosting with TimeSeriesSplit? (y/n): ",
                                                            ["y", "n"]) == "y"),
        "5": pipeline.step_database,
        "6": pipeline.step_sql,
        "7": pipeline.step_benchmarks,
        "8": city_card,
        "9": predict_tomorrow,
        "10": lambda: pipeline.run_all(skip_download=True),
    }
    while True:
        print("\n=== Climate and Weather of Kazakhstan Cities ===")
        for key, text in MENU.items():
            print(f" {key:>2} - {text}")
        choice = ask_choice("Choose an option: ", list(MENU))
        if choice == "0":
            print("Bye!")
            break
        try:
            actions[choice]()
        except FileNotFoundError as exc:
            print(f"  ! {exc}")
        except KeyboardInterrupt:
            print("\n  interrupted, back to menu")
        except Exception as exc:                      # keep the menu alive on unexpected errors
            print(f"  ! step failed: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run the whole pipeline non-interactively")
    parser.add_argument("--skip-download", action="store_true", help="with --all: use already downloaded data")
    parser.add_argument("--no-tune", action="store_true", help="with --all: skip TimeSeriesSplit tuning")
    args = parser.parse_args()
    pd.set_option("display.width", 160)
    if args.all:
        from src import pipeline
        pipeline.run_all(skip_download=args.skip_download, tune=not args.no_tune)
    else:
        menu()


if __name__ == "__main__":
    mp.freeze_support()
    main()
