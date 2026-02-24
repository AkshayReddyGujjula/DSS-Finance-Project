"""
fetch_committees.py
Re-fetches GovTrack committee memberships with corrected yfinance sector labels.
Run once to rebuild raw_committees.csv, then re-run build_dataset.py.
"""
import re, time, requests, pandas as pd

SECTOR_MAP = {
    "Agriculture":            "Consumer Defensive",
    "Armed Services":         "Industrials",
    "Banking":                "Financial Services",
    "Commerce":               "Communication Services",
    "Energy and Commerce":    "Energy",
    "Financial Services":     "Financial Services",
    "Foreign Affairs":        "Industrials",
    "Health":                 "Healthcare",
    "Intelligence":           "Industrials",
    "Judiciary":              "Unknown",
    "Natural Resources":      "Energy",
    "Science and Technology": "Technology",
    "Transportation":         "Industrials",
    "Veterans Affairs":       "Healthcare",
    "Ways and Means":         "Financial Services",
}

def to_sector(name):
    for k, v in SECTOR_MAP.items():
        if k.lower() in str(name).lower():
            return v
    return "Unknown"

def norm(raw):
    s = re.sub(r"^(Sen\.|Rep\.|Del\.)\s*", "", str(raw))
    s = re.sub(r"\s*\[[A-Z-]+\]\s*$", "", s)
    return s.strip().lower()

base_url = "https://www.govtrack.us/api/v2/committee_member"
all_items, offset = [], 0

while True:
    try:
        r = requests.get(base_url, params={"limit": 300, "offset": offset}, timeout=20)
    except Exception as e:
        print(f"Network error at offset {offset}: {e}")
        break
    if r.status_code != 200:
        print(f"HTTP {r.status_code} at offset {offset}")
        break
    items = r.json().get("objects", [])
    if not items:
        break
    all_items.extend(items)
    print(f"  {len(all_items)} committee memberships fetched")
    offset += 300
    time.sleep(0.5)

if not all_items:
    print("No data retrieved.")
    raise SystemExit(1)

rows = []
for item in all_items:
    p = item.get("person", {})
    c = item.get("committee", {})
    member = norm(p.get("name", "") if isinstance(p, dict) else str(p))
    committee = c.get("name", "") if isinstance(c, dict) else str(c)
    rows.append({"member_name": member, "committee_sector": to_sector(committee)})

df = pd.DataFrame(rows).drop_duplicates()
df.to_csv("raw_committees.csv", index=False)
print(f"\nSaved {len(df)} rows to raw_committees.csv")
print("Sector distribution:", df.committee_sector.value_counts().to_dict())
print("Unique members:", df.member_name.nunique())
