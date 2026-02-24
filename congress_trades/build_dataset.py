"""
build_dataset.py
================
Full pipeline to produce congressional_trades_master.csv

Verified working data sources (as of Feb 2026):
  Phase 2 → Quiver Quant public API    → raw_trades.csv
             (free, no key, ~1000 most recent trades)
             OR  pre-downloaded Kaggle CSV (full 2012-2025 history)
  Phase 3 → yfinance sector lookup     → raw_sectors.csv
             (returns already included in Quiver Quant data)
  Phase 4 → Congress.gov API           → raw_members.csv
             (requires free API key — skip gracefully without one)
  Phase 5 → GovTrack API               → raw_committees.csv
             (no key needed, 3,908 memberships confirmed live)
  Phase 6 → pandas merge               → congressional_trades_master.csv

RESUME SUPPORT: Each phase checks for its intermediate CSV before running.
Re-run the script at any time — completed phases are skipped automatically.

─────────────────────────────────────────────────────────────────────────────
SETUP
─────────────────────────────────────────────────────────────────────────────
1. (Optional) Get a free Congress.gov key at https://api.congress.gov/sign-up
   and paste it below as CONGRESS_API_KEY.

2. (Optional, for full 2012-2025 history) Download the Kaggle dataset:
     pip install kaggle
     # Place kaggle.json in ~/.kaggle/
     kaggle datasets download -d shabbarank/congressional-trading-inception-to-march-23
     # Unzip and place the CSV in this folder as "kaggle_trades.csv"

3. Run:  python build_dataset.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
from datetime import timedelta

import requests
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Optional: free key from https://api.congress.gov/sign-up
CONGRESS_API_KEY = "TPwOTZbcjumZb3GQwj6WDBHupThzjMJ4WtHi750i"

# If you have downloaded the Kaggle backup CSV, set this path:
KAGGLE_CSV_PATH = "kaggle_trades.csv"

# Intermediate + final output filenames (written to the current working directory)
OUT_RAW_TRADES     = "raw_trades.csv"
OUT_RAW_SECTORS    = "raw_sectors.csv"
OUT_RAW_MEMBERS    = "raw_members.csv"
OUT_RAW_COMMITTEES = "raw_committees.csv"
OUT_MASTER         = "congressional_trades_master.csv"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_close(df):
    """
    yfinance >=0.2 returns MultiIndex columns when auto_adjust=True.
    Returns a plain Close Series regardless of yfinance version.
    """
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [c for c in df.columns if c[0] == "Close"]
        return df[close_cols[0]] if close_cols else None
    return df["Close"] if "Close" in df.columns else None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Congressional Trades via Quiver Quant (or Kaggle backup)
# ─────────────────────────────────────────────────────────────────────────────
#
# Quiver Quant /live/ endpoint returns the ~1,000 most recent disclosed trades
# with all key fields including pre-computed returns (no API key needed).
#
# Field mapping from Quiver Quant:
#   Representative  -> member_name         House       -> chamber
#   ReportDate      -> disclosure_date     Party       -> party
#   TransactionDate -> trade_date          TickerType  -> asset_type
#   Ticker          -> ticker              Range       -> trade_value_bucket
#   Transaction     -> trade_type
#   PriceChange     -> stock_return_pct    (% since trade date)
#   SPYChange       -> sp500_return_pct
#   ExcessReturn    -> excess_return_pct   (stock minus SPY)
#
# For full 2021-2026 history: use the Kaggle CSV (see SETUP above).
# ─────────────────────────────────────────────────────────────────────────────

QUIVER_URL = "https://api.quiverquant.com/beta/live/congresstrading"


def fetch_quiver_trades():
    print("  Fetching from Quiver Quant /live/congresstrading ...")
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        r = requests.get(QUIVER_URL, headers=headers, timeout=20)
    except requests.RequestException as exc:
        print(f"  Network error: {exc}")
        return pd.DataFrame()

    if r.status_code != 200:
        print(f"  HTTP {r.status_code} from Quiver Quant")
        return pd.DataFrame()

    data = r.json()
    if not isinstance(data, list) or not data:
        print(f"  Unexpected response format: {type(data)}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    print(f"  Quiver Quant returned {len(df)} records")
    return df


def load_kaggle_trades(path=KAGGLE_CSV_PATH):
    print(f"  Loading Kaggle backup from {path} ...")
    # File uses Windows-1252 / latin-1 encoding (contains smart quotes etc.)
    df = pd.read_csv(path, encoding="latin-1")
    print(f"  Kaggle CSV: {len(df)} rows, columns: {df.columns.tolist()}")
    return df


def clean_quiver(df):
    """Normalise Quiver Quant fields to project schema."""
    rename = {
        "Representative":  "member_name",
        "ReportDate":      "disclosure_date",
        "TransactionDate": "trade_date",
        "Ticker":          "ticker",
        "Transaction":     "trade_type",
        "Range":           "trade_value_bucket",
        "House":           "chamber",
        "Party":           "party",
        "TickerType":      "asset_type",
        "Amount":          "amount_usd",
        "PriceChange":     "stock_return_pct",
        "SPYChange":       "sp500_return_pct",
        "ExcessReturn":    "excess_return_pct",
        "Description":     "company",
        "BioGuideID":      "bioguide_id",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["trade_date"]      = pd.to_datetime(df["trade_date"],      errors="coerce")
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
    df["disclosure_lag_days"] = (df["disclosure_date"] - df["trade_date"]).dt.days

    # Convert % values to decimal fractions for consistency
    for col in ("stock_return_pct", "sp500_return_pct", "excess_return_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100

    # Keep stock trades only
    if "asset_type" in df.columns:
        before = len(df)
        df = df[df["asset_type"].str.lower().str.contains("stock", na=False)]
        print(f"  Filtered to stock trades: {before} -> {len(df)} rows")

    return df


def clean_kaggle(df):
    """
    Normalise the Quiver-sourced Kaggle Congressional Trading CSV to project schema.
    Actual columns (verified Feb 2026):
      Ticker, TickerType, Company, Traded, Transaction, Trade_Size_USD,
      Status, Subholding, Description, Name, Filed, Party, District,
      Chamber, Comments, Quiver_Upload_Time, excess_return, State, last_modified
    """
    rename_map = {
        # Kaggle column          → project schema
        "Name":             "member_name",
        "Traded":           "trade_date",      # "Monday, March 11, 2024"
        "Filed":            "disclosure_date",
        "Ticker":           "ticker",
        "Company":          "company",
        "Transaction":      "trade_type",
        "Trade_Size_USD":   "trade_value_bucket",
        "Chamber":          "chamber",
        "Party":            "party",
        "State":            "state",
        "District":         "district",
        "TickerType":       "asset_type",
        "excess_return":    "excess_return_pct_kaggle",  # store separately to track source
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Date parsing — Kaggle stores trade dates as "Monday, March 11, 2024"
    df["trade_date"]      = pd.to_datetime(df["trade_date"],      errors="coerce", format="mixed")
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
    df["disclosure_lag_days"] = (df["disclosure_date"] - df["trade_date"]).dt.days

    # TickerType == 'ST' means equities; filter everything else out
    if "asset_type" in df.columns:
        before = len(df)
        df = df[df["asset_type"].str.upper().str.strip() == "ST"]
        print(f"  Filtered to ST (stock) trades: {before} -> {len(df)} rows")

    # Convert excess_return from percentage points to decimal fraction (consistent with Quiver)
    if "excess_return_pct_kaggle" in df.columns:
        df["excess_return_pct"] = pd.to_numeric(df["excess_return_pct_kaggle"], errors="coerce") / 100
        df.drop(columns=["excess_return_pct_kaggle"], inplace=True)

    return df


def run_phase2():
    print("\n-- Phase 2: Congressional Trades ---------------------------------")

    if os.path.exists(OUT_RAW_TRADES):
        print(f"  Skipping -- {OUT_RAW_TRADES} already exists")
        return pd.read_csv(OUT_RAW_TRADES, parse_dates=["trade_date", "disclosure_date"])

    # Try Quiver Quant (no key needed)
    df_quiver = fetch_quiver_trades()
    if not df_quiver.empty:
        df_quiver = clean_quiver(df_quiver)

    # Load Kaggle CSV if available (for full historical coverage)
    df_kaggle = pd.DataFrame()
    if os.path.exists(KAGGLE_CSV_PATH):
        try:
            df_kaggle = clean_kaggle(load_kaggle_trades())
        except Exception as e:
            print(f"  Warning: could not load Kaggle CSV: {e}")

    if df_quiver.empty and df_kaggle.empty:
        print("\n  No trade data retrieved.")
        print("  To get the full 2012-2025 dataset, download from Kaggle:")
        print("    https://www.kaggle.com/datasets/shabbarank/congressional-trading-inception-to-march-23")
        print("  Save the CSV as 'kaggle_trades.csv' in this folder, then re-run.")
        return pd.DataFrame()

    frames = [f for f in [df_quiver, df_kaggle] if not f.empty]
    df = pd.concat(frames, ignore_index=True)

    subset = [c for c in ("member_name", "ticker", "trade_date", "trade_type") if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    if len(df) < before:
        print(f"  Deduplicated {before - len(df)} overlapping rows")

    df.to_csv(OUT_RAW_TRADES, index=False)
    print(f"  Saved -> {OUT_RAW_TRADES}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — yfinance: sector lookup + fill-in returns for Kaggle rows
# ─────────────────────────────────────────────────────────────────────────────

def get_ticker_info(ticker):
    """Returns (sector, short_name) for a ticker via yfinance."""
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", None) or "Unknown"
        name   = info.get("shortName", None) or info.get("longName", None) or ""
        return sector, name
    except Exception:
        return "Unknown", ""


def get_30d_return(ticker, trade_date):
    """30-day return via yfinance. Used only for rows without pre-computed returns."""
    start = trade_date
    end   = trade_date + timedelta(days=45)
    try:
        stock = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        spy   = yf.download("SPY",  start=start, end=end, progress=False, auto_adjust=True)

        cs   = _safe_close(stock)
        cspy = _safe_close(spy)

        if cs is None or cspy is None or len(cs) < 2 or len(cspy) < 2:
            return float("nan"), float("nan")

        idx    = min(30, len(cs) - 1)
        s_ret  = float(cs.iloc[idx]                        / cs.iloc[0])   - 1
        sp_ret = float(cspy.iloc[min(30, len(cspy) - 1)]  / cspy.iloc[0]) - 1
        return round(s_ret, 6), round(sp_ret, 6)
    except Exception:
        return float("nan"), float("nan")


def run_phase3(df_trades):
    print("\n-- Phase 3: yfinance -- sector lookup & return fill-in -----------")

    # ---- Sector lookup ------------------------------------------------------
    if os.path.exists(OUT_RAW_SECTORS):
        print(f"  Skipping sector fetch -- {OUT_RAW_SECTORS} already exists")
        df_sectors = pd.read_csv(OUT_RAW_SECTORS)
    else:
        unique_tickers = (
            df_trades["ticker"].dropna().unique()
            if "ticker" in df_trades.columns else []
        )
        print(f"  Fetching sectors for {len(unique_tickers)} unique tickers ...")
        rows = []
        for t in tqdm(unique_tickers, desc="  Sectors"):
            sector, name = get_ticker_info(t)
            rows.append({"ticker": t, "sector": sector, "company_name": name})
            time.sleep(0.2)
        df_sectors = pd.DataFrame(rows)
        df_sectors.to_csv(OUT_RAW_SECTORS, index=False)
        print(f"  Saved -> {OUT_RAW_SECTORS}")

    df_trades = df_trades.merge(df_sectors, on="ticker", how="left")

    # Fill blank company names with the name from yfinance
    if "company_name" in df_trades.columns:
        if "company" in df_trades.columns:
            df_trades["company"] = df_trades["company"].fillna(df_trades["company_name"])
        else:
            df_trades["company"] = df_trades["company_name"]
        df_trades.drop(columns=["company_name"], inplace=True)

    # ---- Fill in missing 30d returns (for Quiver rows only) -----------------
    # Kaggle rows already carry excess_return_pct but NOT individual stock/SPY returns.
    # Computing per-row yfinance prices for 46K rows would take days, so we skip
    # those rows and accept NaN for stock_return_pct / sp500_return_pct on Kaggle rows.
    # Only fetch for Quiver rows (recent trades) where stock_return_pct is NaN.
    has_ret = "stock_return_pct" in df_trades.columns

    # A row needs yfinance only if stock_return_pct is missing AND excess_return_pct
    # is also missing (Kaggle rows have excess but no individual components — that's fine).
    if has_ret and "ticker" in df_trades.columns and "trade_date" in df_trades.columns:
        if "excess_return_pct" in df_trades.columns:
            needs_fetch = df_trades["stock_return_pct"].isna() & df_trades["excess_return_pct"].isna()
        else:
            needs_fetch = df_trades["stock_return_pct"].isna()

        pairs = (
            df_trades.loc[needs_fetch, ["ticker", "trade_date"]]
            .dropna()
            .drop_duplicates()
        )
        if len(pairs) > 0:
            print(f"  Computing 30-day returns for {len(pairs)} pairs missing both return columns ...")
            ret_cache = {}
            for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="  Returns"):
                key = (row["ticker"], row["trade_date"])
                ret_cache[key] = get_30d_return(row["ticker"], row["trade_date"])
                time.sleep(0.25)

            def _fill(col_idx, row):
                key = (row.get("ticker"), row.get("trade_date"))
                if pd.notna(row.get("stock_return_pct")):
                    return row["stock_return_pct"] if col_idx == 0 else row.get("sp500_return_pct")
                return ret_cache.get(key, (float("nan"), float("nan")))[col_idx]

            if ret_cache:
                df_trades["stock_return_pct"] = df_trades.apply(lambda r: _fill(0, r), axis=1)
                df_trades["sp500_return_pct"] = df_trades.apply(lambda r: _fill(1, r), axis=1)
        else:
            print("  No rows need yfinance return computation -- all have at least one return column")
    else:
        print("  Skipping yfinance return fetch (no stock_return_pct column yet)")
        if not has_ret:
            df_trades["stock_return_pct"] = float("nan")
            df_trades["sp500_return_pct"] = float("nan")

    # Compute / fill excess return (stock minus S&P 500)
    if "stock_return_pct" in df_trades.columns and "sp500_return_pct" in df_trades.columns:
        computed = df_trades["stock_return_pct"] - df_trades["sp500_return_pct"]
        if "excess_return_pct" in df_trades.columns:
            # Fill NaN values in the existing column (e.g. Quiver Quant nulls)
            df_trades["excess_return_pct"] = df_trades["excess_return_pct"].fillna(computed)
        else:
            df_trades["excess_return_pct"] = computed

    return df_trades


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Congress.gov: member metadata (optional)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase4():
    print("\n-- Phase 4: Congress.gov member metadata -------------------------")

    if os.path.exists(OUT_RAW_MEMBERS):
        print(f"  Skipping -- {OUT_RAW_MEMBERS} already exists")
        return pd.read_csv(OUT_RAW_MEMBERS)

    if CONGRESS_API_KEY == "YOUR_KEY_HERE":
        print("  CONGRESS_API_KEY not set -- skipping Phase 4.")
        print("  Party/chamber data already comes from Quiver Quant, so this is optional.")
        print("  For richer state/district metadata register free at https://api.congress.gov/sign-up")
        return pd.DataFrame()

    all_members = []
    for congress_num in range(117, 120):   # 117th-119th (2021-2026)
        offset = 0
        while True:
            url    = f"https://api.congress.gov/v3/member/{congress_num}"
            params = {"limit": 250, "offset": offset, "api_key": CONGRESS_API_KEY}
            try:
                r = requests.get(url, params=params, timeout=15)
            except requests.RequestException as exc:
                print(f"  Network error (congress {congress_num}): {exc}")
                break

            if r.status_code != 200:
                print(f"  HTTP {r.status_code} at congress {congress_num}, offset {offset}")
                break

            data = r.json().get("members", [])
            if not data:
                break

            all_members.extend(data)
            print(f"  Congress {congress_num}: {len(all_members)} members total")
            offset += 250
            time.sleep(0.5)

    if not all_members:
        return pd.DataFrame()

    df = pd.json_normalize(all_members)
    rename_map = {
        "name":      "member_name",
        "partyName": "party_meta",
        "party":     "party_meta",
        "state":     "state",
        "chamber":   "chamber_meta",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    keep = [c for c in ("member_name", "party_meta", "state", "chamber_meta") if c in df.columns]
    df   = df[keep].drop_duplicates(subset=["member_name"])

    df.to_csv(OUT_RAW_MEMBERS, index=False)
    print(f"  Saved -> {OUT_RAW_MEMBERS}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — GovTrack: committee memberships (confirmed working Feb 2026)
# ─────────────────────────────────────────────────────────────────────────────

# Sector labels MUST match yfinance sector strings exactly.
# yfinance uses: "Financial Services", "Consumer Defensive", "Technology", etc.
COMMITTEE_SECTOR_MAP = {
    "Agriculture":            "Consumer Defensive",      # yfinance: Consumer Defensive
    "Armed Services":         "Industrials",
    "Banking":                "Financial Services",       # yfinance: Financial Services
    "Commerce":               "Communication Services",
    "Energy and Commerce":    "Energy",
    "Financial Services":     "Financial Services",       # yfinance: Financial Services
    "Foreign Affairs":        "Industrials",
    "Health":                 "Healthcare",
    "Intelligence":           "Industrials",
    "Judiciary":              "Unknown",
    "Natural Resources":      "Energy",
    "Science and Technology": "Technology",
    "Transportation":         "Industrials",
    "Veterans Affairs":       "Healthcare",
    "Ways and Means":         "Financial Services",       # yfinance: Financial Services
}


def _committee_to_sector(name):
    if not isinstance(name, str):
        return "Unknown"
    for keyword, sector in COMMITTEE_SECTOR_MAP.items():
        if keyword.lower() in name.lower():
            return sector
    return "Unknown"


def run_phase5():
    print("\n-- Phase 5: GovTrack committee memberships -----------------------")

    if os.path.exists(OUT_RAW_COMMITTEES):
        print(f"  Skipping -- {OUT_RAW_COMMITTEES} already exists")
        df = pd.read_csv(OUT_RAW_COMMITTEES)
        lookup = df.groupby("member_name")["committee_sector"].apply(set).to_dict()
        print(f"  Loaded {len(lookup)} members from cache")
        return df, lookup

    base_url  = "https://www.govtrack.us/api/v2/committee_member"
    all_items = []
    offset    = 0

    while True:
        try:
            r = requests.get(base_url, params={"limit": 300, "offset": offset}, timeout=20)
        except requests.RequestException as exc:
            print(f"  Network error at offset {offset}: {exc}")
            break

        if r.status_code != 200:
            print(f"  HTTP {r.status_code} at offset {offset}")
            break

        items = r.json().get("objects", [])
        if not items:
            break

        all_items.extend(items)
        print(f"  {len(all_items)} committee memberships fetched")
        offset += 300
        time.sleep(0.5)

    if not all_items:
        print("  Warning: no committee data retrieved -- aligned_trade will be 0 for all rows")
        return pd.DataFrame(), {}

    df_raw = pd.DataFrame(all_items)
    df_raw["member_name_full"] = df_raw["person"].apply(
        lambda x: x.get("name", "") if isinstance(x, dict) else str(x)
    )
    df_raw["committee_name"] = df_raw["committee"].apply(
        lambda x: x.get("name", "") if isinstance(x, dict) else str(x)
    )
    df_raw["committee_sector"] = df_raw["committee_name"].apply(_committee_to_sector)

    # GovTrack names look like "Sen. John Boozman [R-AR]" or "Rep. Nancy Pelosi [D-CA]"
    # Strip title and party/state bracket so we match Quiver Quant's "First Last" format
    import re as _re
    def _normalise_gov_name(raw):
        s = _re.sub(r'^(Sen\.|Rep\.|Del\.)\s*', '', str(raw))   # remove title
        s = _re.sub(r'\s*\[[A-Z-]+\]\s*$', '', s)               # remove [R-AR]
        return s.strip().lower()

    df_raw["member_name"] = df_raw["member_name_full"].apply(_normalise_gov_name)

    df = df_raw[["member_name", "committee_sector"]].drop_duplicates()
    df.to_csv(OUT_RAW_COMMITTEES, index=False)
    print(f"  Saved -> {OUT_RAW_COMMITTEES}  ({len(df)} rows)")

    lookup = df.groupby("member_name")["committee_sector"].apply(set).to_dict()
    print(f"  Committee-sector lookup built for {len(lookup)} members")
    return df, lookup


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Merge all datasets
# ─────────────────────────────────────────────────────────────────────────────

def run_phase6(df_trades, df_members, committee_sector_lookup):
    print("\n-- Phase 6: Merging all datasets ---------------------------------")

    df = df_trades.copy()

    # Merge Congress.gov member metadata (optional enrichment)
    if (not df_members.empty
            and "member_name" in df_members.columns
            and "member_name" in df.columns):
        df["_key"]         = df["member_name"].str.strip().str.lower()
        df_m               = df_members.copy()
        df_m["_key"]       = df_m["member_name"].str.strip().str.lower()
        meta_cols          = ["_key"] + [c for c in ("party_meta", "state", "chamber_meta")
                                          if c in df_m.columns]
        df = df.merge(df_m[meta_cols], on="_key", how="left")
        df.drop(columns=["_key"], inplace=True)

        # Fill blanks in existing party/chamber columns
        for src, dst in (("party_meta", "party"), ("chamber_meta", "chamber")):
            if src in df.columns and dst in df.columns:
                df[dst] = df[dst].fillna(df[src])
        df.drop(columns=[c for c in ("party_meta", "chamber_meta") if c in df.columns],
                inplace=True)

    print(f"  Rows after member merge: {len(df)}")

    # aligned_trade flag: 1 if trade sector matches a committee the member sits on
    # Normalise trade member name the same way as GovTrack names were normalised
    import re as _re
    def _norm(name):
        s = _re.sub(r'^(Sen\.|Rep\.|Del\.)\s*', '', str(name))
        s = _re.sub(r'\s*\[[A-Z-]+\]\s*$', '', s)
        return s.strip().lower()

    def is_aligned(row):
        key = _norm(row.get("member_name", ""))
        sectors = committee_sector_lookup.get(key, set())
        return 1 if row.get("sector", "Unknown") in sectors else 0

    df["aligned_trade"] = df.apply(is_aligned, axis=1)
    n = df["aligned_trade"].sum()
    print(f"  aligned_trade: {n} aligned ({100 * n / max(len(df), 1):.1f}%)")

    # Final column ordering (extras appended at end)
    ordered = [
        "trade_date", "disclosure_date", "disclosure_lag_days",
        "member_name", "bioguide_id", "chamber", "party", "state",
        "ticker", "company", "sector",
        "trade_type", "trade_value_bucket", "trade_value_est", "amount_usd",
        "stock_return_pct", "sp500_return_pct", "excess_return_pct", "beat_market",
        "aligned_trade",
    ]
    present = [c for c in ordered if c in df.columns]
    extras  = [c for c in df.columns if c not in ordered]
    return df[present + extras]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Final quality pass & analysis-ready columns
# ─────────────────────────────────────────────────────────────────────────────

# Trade-value bucket → numeric midpoint estimate (USD).
# Used when amount_usd is not available (Kaggle rows).
BUCKET_MIDPOINTS = {
    "$1,001 - $15,000":           8_001,
    "$15,001 - $50,000":         32_500,
    "$50,001 - $100,000":        75_000,
    "$100,001 - $250,000":      175_000,
    "$250,001 - $500,000":      375_000,
    "$500,001 - $1,000,000":    750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
    "$5,000,001 - $25,000,000": 15_000_000,
    "$25,000,001 - $50,000,000": 37_500_000,
    "$50,000,001+":              75_000_000,
}

# Partial / malformed bucket strings → canonical form
BUCKET_FIXES = {
    "$1,001":       "$1,001 - $15,000",
    "$15,001":      "$15,001 - $50,000",
    "$50,001":      "$50,001 - $100,000",
    "$100,001":     "$100,001 - $250,000",
    "$250,001":     "$250,001 - $500,000",
    "$500,001":     "$500,001 - $1,000,000",
    "$1,000,001":   "$1,000,001 - $5,000,000",
    "$5,000,001":   "$5,000,001 - $25,000,000",
}


def run_finalize(df):
    """
    Final data quality pass and addition of analysis-ready columns.
    Called after Phase 6 (merge) and before saving the master CSV.

    Operations:
      1. Drop rows with negative disclosure_lag_days (data entry errors).
      2. Normalise malformed trade_value_bucket strings.
      3. Add trade_value_est  — numeric midpoint of the bucket (or amount_usd).
      4. Add beat_market      — binary target: 1 if excess_return_pct > 0.
      5. Winsorise excess_return_pct at the 1st/99th percentile to remove
         extreme data-quality outliers while keeping genuine large moves.
      6. Ensure sector/company are clean strings.
    """
    print("\n-- Phase 7: Final cleanup & analysis columns -------------------")
    n_start = len(df)

    # 1. Remove rows where disclosure happened before the trade (impossible)
    if "disclosure_lag_days" in df.columns:
        bad = df["disclosure_lag_days"] < 0
        if bad.sum() > 0:
            print(f"  Dropping {bad.sum()} rows with negative disclosure_lag_days")
            df = df[~bad].copy()

    # 2. Fix malformed trade_value_bucket strings
    if "trade_value_bucket" in df.columns:
        before_fix = df["trade_value_bucket"].isin(BUCKET_FIXES).sum()
        df["trade_value_bucket"] = df["trade_value_bucket"].replace(BUCKET_FIXES)
        if before_fix:
            print(f"  Fixed {before_fix} malformed trade_value_bucket values")

    # 3. Add trade_value_est (numeric midpoint)
    if "trade_value_bucket" in df.columns:
        df["trade_value_est"] = df["trade_value_bucket"].map(BUCKET_MIDPOINTS)
        # If we have a precise amount_usd, use that instead
        if "amount_usd" in df.columns:
            df["trade_value_est"] = (
                pd.to_numeric(df["amount_usd"], errors="coerce")
                .combine_first(df["trade_value_est"])
            )
        filled = df["trade_value_est"].notna().sum()
        print(f"  trade_value_est: {filled:,}/{len(df):,} rows filled")

    # 4. Binary target variable: beat the market?
    if "excess_return_pct" in df.columns:
        df["beat_market"] = (df["excess_return_pct"] > 0).astype("Int8")
        df.loc[df["excess_return_pct"].isna(), "beat_market"] = pd.NA
        pos = (df["beat_market"] == 1).sum()
        total = df["beat_market"].notna().sum()
        print(f"  beat_market: {pos:,}/{total:,} trades beat market ({100*pos/max(total,1):.1f}%)")

    # 5. Winsorise excess_return_pct (removes data-quality extremes only)
    if "excess_return_pct" in df.columns:
        lo = df["excess_return_pct"].quantile(0.01)
        hi = df["excess_return_pct"].quantile(0.99)
        n_out = ((df["excess_return_pct"] < lo) | (df["excess_return_pct"] > hi)).sum()
        df["excess_return_pct"] = df["excess_return_pct"].clip(lower=lo, upper=hi)
        print(f"  Winsorised {n_out} extreme excess_return_pct values "
              f"(p1={lo:.3f}, p99={hi:.3f})")

    # 6. Tidy string columns
    for col in ("sector", "company", "chamber", "party", "state"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown" if col in ("sector",) else "").str.strip()
            if col == "sector":
                df[col] = df[col].replace("", "Unknown")

    print(f"  Rows: {n_start:,} -> {len(df):,} (net removed {n_start - len(df)})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  Congressional Trades -- Master Dataset Builder")
    print("=" * 66)

    df_trades = run_phase2()

    if df_trades.empty:
        print("\n  Cannot continue without trade data. See instructions above.")
        return

    print(f"\n  Trades loaded: {len(df_trades)} rows, {len(df_trades.columns)} columns")
    print(f"  Columns: {df_trades.columns.tolist()}")

    df_trades  = run_phase3(df_trades)
    df_members = run_phase4()
    _, committee_sector_lookup = run_phase5()

    df_final = run_phase6(df_trades, df_members, committee_sector_lookup)
    df_final = run_finalize(df_final)
    df_final.to_csv(OUT_MASTER, index=False)

    print("\n" + "=" * 66)
    print(f"  DONE -- {len(df_final)} rows saved to {OUT_MASTER}")
    print("=" * 66)

    print("\nColumn types:")
    print(df_final.dtypes.to_string())
    print("\nNumeric summary:")
    print(df_final.describe().to_string())
    print("\nMissing values per column:")
    print(df_final.isnull().sum().to_string())
    print("\nFirst 5 rows:")
    print(df_final.head().to_string())


if __name__ == "__main__":
    main()
