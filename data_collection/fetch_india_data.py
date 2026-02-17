import argparse
import datetime
import os
import shutil
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm

from qlib_dump_bin import DumpDataAll


DEFAULT_NSE_SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]

DEFAULT_NSE_SYMBOLS_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def _load_symbols_from_nse_archive(symbols_url: str, timeout_s: int = 20) -> List[str]:
    print(f"Loading NSE symbols from URL: {symbols_url} (timeout={timeout_s}s)", flush=True)
    try:
        req = Request(symbols_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(raw))
    except Exception as exc:
        print(f"[WARN] failed to read NSE symbol list from {symbols_url}: {exc}", flush=True)
        return DEFAULT_NSE_SYMBOLS

    if "SYMBOL" not in df.columns:
        print(f"[WARN] NSE symbol list missing 'SYMBOL' column, fallback to defaults: {symbols_url}", flush=True)
        return DEFAULT_NSE_SYMBOLS

    symbols = sorted(df["SYMBOL"].dropna().astype(str).str.strip().unique())
    symbols = [f"{sym}.NS" for sym in symbols if sym]
    if not symbols:
        print(f"[WARN] NSE symbol list empty, fallback to defaults: {symbols_url}", flush=True)
        return DEFAULT_NSE_SYMBOLS

    print(f"Loaded {len(symbols)} symbols from NSE archive", flush=True)
    return symbols


def _sanitize_symbol_for_qlib(yf_symbol: str) -> str:
    """Convert yfinance symbol like RELIANCE.NS to qlib-safe code like NSRELIANCE."""
    raw = yf_symbol.upper().replace(".NS", "")
    raw = "".join(ch for ch in raw if ch.isalnum())
    return f"NS{raw}"


def _read_symbols(symbols_file: Optional[str], symbols_url: str, symbols_url_timeout: int) -> List[str]:
    if symbols_file is None:
        return _load_symbols_from_nse_archive(symbols_url, timeout_s=symbols_url_timeout)

    p = Path(os.path.expanduser(symbols_file))
    if not p.exists():
        raise FileNotFoundError(f"symbols_file not found: {p}")

    print(f"Loading symbols from local file: {p}", flush=True)
    if p.suffix.lower() in {".csv"}:
        df = pd.read_csv(p)
        if "symbol" in df.columns:
            symbols = df["symbol"].dropna().astype(str).tolist()
        else:
            symbols = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        symbols = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    return symbols


def _ensure_dirs(base_path: str) -> Dict[str, Path]:
    base = Path(os.path.expanduser(base_path))
    dirs = {
        "base": base,
        "k_data": base / "k_data",
        "export": base / "export",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _download_one_symbol(
    yf_symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for NSE data collection. Install it with `pip install yfinance`."
        ) from exc

    df = yf.download(
        yf_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        interval="1d",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Handle possible multi-index columns returned by yfinance.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{yf_symbol}: missing required columns from yfinance output: {missing}")

    out = pd.DataFrame(index=pd.to_datetime(df.index))
    out["open"] = df["Open"].astype(float)
    out["high"] = df["High"].astype(float)
    out["low"] = df["Low"].astype(float)
    out["close"] = df["Close"].astype(float)
    out["volume"] = df["Volume"].fillna(0).astype(float)
    out["amount"] = out["close"] * out["volume"]
    out["vwap"] = out["amount"] / out["volume"].replace(0, pd.NA)
    out["vwap"] = out["vwap"].fillna(out["close"])
    out["factor"] = 1.0
    out.index.name = "date"
    return out


def build_nse_qlib_data(
    save_path: str,
    qlib_export_path: str,
    symbols_file: Optional[str] = None,
    symbols_url: str = DEFAULT_NSE_SYMBOLS_URL,
    symbols_url_timeout: int = 20,
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    max_workers: int = 8,
):
    dirs = _ensure_dirs(save_path)
    qlib_dir = Path(os.path.expanduser(qlib_export_path))
    qlib_dir.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    symbols = _read_symbols(symbols_file, symbols_url, symbols_url_timeout)
    symbol_rows = []

    print(f"Downloading {len(symbols)} NSE symbols from yfinance", flush=True)
    for yf_symbol in tqdm(symbols):
        try:
            df = _download_one_symbol(yf_symbol, start_date=start_date, end_date=end_date)
        except Exception as exc:
            print(f"[WARN] skip {yf_symbol}: {exc}", flush=True)
            continue

        if df.empty:
            print(f"[WARN] no rows for {yf_symbol}", flush=True)
            continue

        qlib_code = _sanitize_symbol_for_qlib(yf_symbol)
        df["code"] = qlib_code

        df.to_pickle(dirs["k_data"] / f"{qlib_code}.pkl")
        df.to_csv(dirs["export"] / f"{qlib_code}.csv")

        symbol_rows.append(
            {
                "yf_symbol": yf_symbol,
                "code": qlib_code,
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "rows": len(df),
            }
        )

    if not symbol_rows:
        raise RuntimeError("No NSE symbols were exported. Check symbol list and network access.")

    pd.DataFrame(symbol_rows).to_csv(dirs["base"] / "symbol_map.csv", index=False)

    DumpDataAll(
        csv_path=str(dirs["export"]),
        qlib_dir=str(qlib_dir),
        max_workers=max_workers,
        exclude_fields="date,code",
        symbol_field_name="code",
    ).dump()

    calendars = qlib_dir / "calendars"
    day_path = calendars / "day.txt"
    day_future = calendars / "day_future.txt"
    if day_path.exists():
        shutil.copy(day_path, day_future)

    print("Done. Generated Qlib dataset:", flush=True)
    print(f"- Raw cache: {dirs['base']}", flush=True)
    print(f"- Qlib output: {qlib_dir}", flush=True)
    print(f"- Symbols exported: {len(symbol_rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch NSE stocks and dump into Qlib format.")
    parser.add_argument("--save_path", default="~/.qlib/tmp_nse")
    parser.add_argument("--qlib_export_path", default="~/.qlib/qlib_data/in_data_rolling")
    parser.add_argument("--symbols_file", default=None, help="Optional txt/csv with symbols like RELIANCE.NS")
    parser.add_argument("--symbols_url", default=DEFAULT_NSE_SYMBOLS_URL,
                        help="NSE CSV URL with SYMBOL column (used when --symbols_file is omitted)")
    parser.add_argument("--symbols_url_timeout", type=int, default=20,
                        help="Timeout in seconds when downloading NSE symbols URL")
    parser.add_argument("--start_date", default="2010-01-01")
    parser.add_argument("--end_date", default=None, help="YYYY-MM-DD, exclusive for yfinance")
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    build_nse_qlib_data(
        save_path=args.save_path,
        qlib_export_path=args.qlib_export_path,
        symbols_file=args.symbols_file,
        symbols_url=args.symbols_url,
        symbols_url_timeout=args.symbols_url_timeout,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
