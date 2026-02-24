"""verify_final.py — one-shot verification of congressional_trades_master.csv"""
import pandas as pd

df = pd.read_csv(
    "congressional_trades_master.csv",
    parse_dates=["trade_date", "disclosure_date"],
    low_memory=False,
)

print("=" * 62)
print("  FINAL VERIFICATION — congressional_trades_master.csv")
print("=" * 62)
print(f"  Shape: {len(df):,} rows x {len(df.columns)} columns")

objectives = {
    "Obj1 – Beat market test":    ["excess_return_pct", "beat_market", "trade_date"],
    "Obj2 – Party comparison":    ["party", "chamber", "excess_return_pct", "sector"],
    "Obj3 – Committee alignment": ["aligned_trade", "excess_return_pct", "sector", "member_name"],
    "Obj4 – Feature importance":  ["party", "aligned_trade", "trade_value_est", "sector", "beat_market"],
    "Obj5 – ML classifier":       ["beat_market", "party", "chamber", "trade_value_est",
                                    "sector", "aligned_trade", "disclosure_lag_days"],
    "Disclosure latency":         ["disclosure_lag_days", "trade_date"],
    "Sector heatmap":             ["sector", "state", "member_name"],
}
print()
print("  OBJECTIVE READINESS:")
all_ok = True
for obj, cols in objectives.items():
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"    FAIL  {obj}: missing columns {missing}")
        all_ok = False
        continue
    worse_null = max(df[c].isna().mean() for c in cols)
    flag = "WARN" if worse_null > 0.10 else "OK  "
    print(f"    {flag}  {obj}")
    for c in cols:
        pct = df[c].isna().mean() * 100
        if pct > 5:
            print(f"           {c}: {pct:.0f}% null (noted)")

print()
print("  KEY STATS:")
print(f"    date range:          {df.trade_date.min().date()} to {df.trade_date.max().date()}")
print(f"    unique members:      {df.member_name.nunique()}")
print(f"    unique tickers:      {df.ticker.nunique()}")
print(f"    chamber:             {df.chamber.value_counts().to_dict()}")
print(f"    party:               {df.party.value_counts().to_dict()}")
beat_counts = df.beat_market.value_counts().to_dict()
total_bm = df.beat_market.notna().sum()
beat_pct  = 100 * beat_counts.get(1, 0) / max(total_bm, 1)
print(f"    beat_market:         {beat_counts}  ({beat_pct:.1f}% beat SPY)")
print(f"    aligned_trade:       {df.aligned_trade.sum():,} / {len(df):,} ({df.aligned_trade.mean()*100:.1f}%)")
print(f"    excess_return_pct:   mean={df.excess_return_pct.mean():.4f}  "
      f"std={df.excess_return_pct.std():.4f}  "
      f"coverage={df.excess_return_pct.notna().mean()*100:.1f}%")
print(f"    disclosure_lag:      mean={df.disclosure_lag_days.mean():.0f}d  "
      f"median={df.disclosure_lag_days.median():.0f}d  max={df.disclosure_lag_days.max()}d")
print(f"    neg lags:            {(df.disclosure_lag_days < 0).sum()} (should be 0)")
print(f"    trade_value_est:     {df.trade_value_est.notna().mean()*100:.1f}% filled")

print()
print("  KNOWN LIMITATIONS:")
senate_n = (df.chamber == "Senate").sum()
print(f"    Senate data:         {senate_n:,} rows ({100*senate_n/len(df):.1f}%) — Kaggle skews toward House")
un_sect = (df.sector == "Unknown").sum()
print(f"    Unknown sector:      {un_sect:,} rows ({100*un_sect/len(df):.1f}%) — yfinance lookup gap")
print(f"    stock/SPY returns:   only {df.stock_return_pct.notna().sum():,} rows — use excess_return_pct instead")
print(f"    aligned_trade:       built from 100 members (GovTrack API offset limit)")
print(f"    Junk columns in CSV: Status, Subholding, Description, Comments (keep or drop before modelling)")

print()
verdict = (
    (df.disclosure_lag_days < 0).sum() == 0
    and "beat_market"    in df.columns
    and "trade_value_est" in df.columns
    and "aligned_trade"  in df.columns
    and df.aligned_trade.sum() > 0
    and all_ok
)
print("  VERDICT:", "✅  READY FOR ANALYSIS" if verdict else "❌  ISSUES REMAIN")
