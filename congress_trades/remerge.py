"""
remerge.py
==========
Fast re-run of Phase 6 + Phase 7 only, using cached intermediate CSVs.
Skips all network calls. Use this after re-running fetch_committees.py
or any time you only want to change the merge/cleanup logic.

Requires: raw_trades.csv, raw_sectors.csv, raw_committees.csv
Produces: congressional_trades_master.csv (overwritten)
"""
import re
import pandas as pd

# ---- Load intermediates ------------------------------------------------
print("Loading cached intermediates ...")

df = pd.read_csv(
    "raw_trades.csv",
    parse_dates=["trade_date", "disclosure_date"],
    low_memory=False,
)
print(f"  raw_trades:      {len(df):,} rows")

df_sectors = pd.read_csv("raw_sectors.csv")
print(f"  raw_sectors:     {len(df_sectors):,} tickers")

df_committees = pd.read_csv("raw_committees.csv")
lookup = df_committees.groupby("member_name")["committee_sector"].apply(set).to_dict()
print(f"  raw_committees:  {len(lookup):,} members in committee lookup")

# ---- Merge sectors ------------------------------------------------------
df = df.merge(df_sectors, on="ticker", how="left")

if "company_name" in df.columns:
    if "company" in df.columns:
        df["company"] = df["company"].fillna(df["company_name"])
    else:
        df["company"] = df["company_name"]
    df.drop(columns=["company_name"], inplace=True)

# Fill sector column
if "sector_x" in df.columns and "sector_y" in df.columns:
    df["sector"] = df["sector_x"].fillna(df["sector_y"])
    df.drop(columns=["sector_x", "sector_y"], inplace=True)
elif "sector_y" in df.columns:
    df["sector"] = df["sector_y"]
    df.drop(columns=["sector_y"], inplace=True)

df["sector"] = df["sector"].fillna("Unknown")

# ---- Aligned trade flag -------------------------------------------------
def _norm(name):
    s = re.sub(r"^(Sen\.|Rep\.|Del\.)\s*", "", str(name))
    s = re.sub(r"\s*\[[A-Z-]+\]\s*$", "", s)
    return s.strip().lower()

def is_aligned(row):
    key = _norm(row.get("member_name", ""))
    sectors = lookup.get(key, set())
    return 1 if row.get("sector", "Unknown") in sectors else 0

df["aligned_trade"] = df.apply(is_aligned, axis=1)
n = df["aligned_trade"].sum()
print(f"\naligned_trade: {n:,} aligned ({100 * n / max(len(df), 1):.1f}%)")

# ---- Phase 7: finalize --------------------------------------------------
BUCKET_MIDPOINTS = {
    "$1,001 - $15,000":            8_001,
    "$15,001 - $50,000":          32_500,
    "$50,001 - $100,000":         75_000,
    "$100,001 - $250,000":       175_000,
    "$250,001 - $500,000":       375_000,
    "$500,001 - $1,000,000":     750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
    "$5,000,001 - $25,000,000": 15_000_000,
    "$25,000,001 - $50,000,000": 37_500_000,
    "$50,000,001+":              75_000_000,
}
BUCKET_FIXES = {
    "$1,001":     "$1,001 - $15,000",
    "$15,001":    "$15,001 - $50,000",
    "$50,001":    "$50,001 - $100,000",
    "$100,001":   "$100,001 - $250,000",
    "$250,001":   "$250,001 - $500,000",
    "$500,001":   "$500,001 - $1,000,000",
    "$1,000,001": "$1,000,001 - $5,000,000",
    "$5,000,001": "$5,000,001 - $25,000,000",
}

n_start = len(df)

# Drop negative lags
bad = df["disclosure_lag_days"] < 0
if bad.sum():
    print(f"Dropping {bad.sum()} negative-lag rows")
    df = df[~bad].copy()

# Fix malformed buckets
df["trade_value_bucket"] = df["trade_value_bucket"].replace(BUCKET_FIXES)

# trade_value_est
df["trade_value_est"] = df["trade_value_bucket"].map(BUCKET_MIDPOINTS)
if "amount_usd" in df.columns:
    df["trade_value_est"] = (
        pd.to_numeric(df["amount_usd"], errors="coerce")
        .combine_first(df["trade_value_est"])
    )

# beat_market
if "excess_return_pct" in df.columns:
    df["beat_market"] = (df["excess_return_pct"] > 0).astype("Int8")
    df.loc[df["excess_return_pct"].isna(), "beat_market"] = pd.NA
    pos   = (df["beat_market"] == 1).sum()
    total = df["beat_market"].notna().sum()
    print(f"beat_market: {pos:,}/{total:,} ({100*pos/max(total,1):.1f}%) beat market")

# Winsorise
if "excess_return_pct" in df.columns:
    lo = df["excess_return_pct"].quantile(0.01)
    hi = df["excess_return_pct"].quantile(0.99)
    n_out = ((df["excess_return_pct"] < lo) | (df["excess_return_pct"] > hi)).sum()
    df["excess_return_pct"] = df["excess_return_pct"].clip(lo, hi)
    print(f"Winsorised {n_out} extreme values (p1={lo:.3f}, p99={hi:.3f})")

# Tidy strings
for col in ("sector", "company", "chamber", "party", "state"):
    if col in df.columns:
        df[col] = df[col].fillna("Unknown" if col == "sector" else "").str.strip()
        if col == "sector":
            df[col] = df[col].replace("", "Unknown")

print(f"\nRows: {n_start:,} -> {len(df):,}")

# ---- Column ordering ----------------------------------------------------
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
df = df[present + extras]

df.to_csv("congressional_trades_master.csv", index=False)
print(f"\nSaved {len(df):,} rows to congressional_trades_master.csv")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\naligned_trade distribution: {df.aligned_trade.value_counts().to_dict()}")
print(f"beat_market distribution:   {dict(df.beat_market.value_counts())}")
print(f"sector distribution:\n{df.sector.value_counts().head(12).to_string()}")
