"""
EDA Analysis - Congressional Stock Trading Performance
=======================================================
Week 3 Deliverable: 8-12 Visualizations + Statistical Analysis

This script creates comprehensive EDA for analyzing Congressional trading performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
DATA_PATH = "congressional_trades_master.csv"
OUTPUT_DIR = "analysis_output/"

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CONGRESSIONAL TRADING EDA ANALYSIS")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["trade_date", "disclosure_date"], low_memory=False)
print(f"    Dataset shape: {df.shape}")

# ============================================================================
# VISUALIZATION 1: Overall Market Outperformance
# ============================================================================
print("\n[Creating Viz 1: Overall Market Performance...]")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1a: Distribution of excess returns
ax1 = axes[0]
valid_returns = df['excess_return_pct'].dropna()
ax1.hist(valid_returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Market (SPY)')
ax1.axvline(x=valid_returns.mean(), color='green', linestyle='-', linewidth=2, 
            label=f'Mean: {valid_returns.mean():.2%}')
ax1.set_xlabel('Excess Return (vs S&P 500)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Congressional Trade Returns')
ax1.legend()

# 1b: Beat market pie chart
ax2 = axes[1]
beat_counts = df['beat_market'].value_counts()
colors = ['#ff6b6b', '#4ecdc4']
labels = ['Underperformed\nMarket', 'Beat Market']
sizes = [beat_counts.get(0, 0), beat_counts.get(1, 0)]
ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
        explode=(0, 0.05), shadow=True)
ax2.set_title('Trades That Beat vs Underperformed SPY')

# 1c: Box plot of returns by outcome
ax3 = axes[2]
df['return_category'] = pd.cut(df['excess_return_pct'], 
                               bins=[-1, -0.1, 0, 0.1, 0.5, float('inf')],
                               labels=['<-10%', '-10% to 0%', '0% to 10%', '10% to 50%', '>50%'])
return_cats = df['return_category'].value_counts().sort_index()
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(return_cats)))
bars = ax3.bar(return_cats.index.astype(str), return_cats.values, color=colors, edgecolor='black')
ax3.set_xlabel('Return Category')
ax3.set_ylabel('Number of Trades')
ax3.set_title('Trade Returns by Category')
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}1_overall_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}1_overall_performance.png")

# ============================================================================
# VISUALIZATION 2: Party Comparison (Democrats vs Republicans)
# ============================================================================
print("[Creating Viz 2: Party Comparison - Democrats vs Republicans...]")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Filter to D and R only
df_party = df[df['party'].isin(['D', 'R'])].copy()

# 2a: Mean excess return by party
ax1 = axes[0, 0]
party_stats = df_party.groupby('party')['excess_return_pct'].agg(['mean', 'std', 'count'])
party_stats = party_stats.reset_index()
colors = {'D': '#3182bd', 'R': '#e6550d'}
bars = ax1.bar(party_stats['party'], party_stats['mean'], 
               yerr=party_stats['std']/np.sqrt(party_stats['count']),
               color=[colors[p] for p in party_stats['party']], 
               edgecolor='black', capsize=5)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax1.set_xlabel('Party')
ax1.set_ylabel('Mean Excess Return')
ax1.set_title('Mean Excess Return by Party')
for i, (idx, row) in enumerate(party_stats.iterrows()):
    ax1.annotate(f'{row["mean"]:.2%}\n(n={row["count"]:,})', 
                 xy=(i, row['mean']), ha='center', va='bottom', fontsize=10)

# 2b: Beat market rate by party
ax2 = axes[0, 1]
party_beat = df_party.groupby('party')['beat_market'].mean() * 100
bars = ax2.bar(party_beat.index, party_beat.values, 
               color=[colors[p] for p in party_beat.index], edgecolor='black')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50% (Random)')
ax2.set_xlabel('Party')
ax2.set_ylabel('Beat Market Rate (%)')
ax2.set_title('Percentage of Trades Beating Market by Party')
ax2.legend()
for i, (party, rate) in enumerate(party_beat.items()):
    ax2.annotate(f'{rate:.1f}%', xy=(i, rate), ha='center', va='bottom', fontsize=11)

# 2c: Distribution of returns by party (violin plot)
ax3 = axes[1, 0]
df_party_plot = df_party[df_party['excess_return_pct'].notna()].copy()
parts = ax3.violinplot([df_party_plot[df_party_plot['party'] == 'D']['excess_return_pct'].dropna(),
                        df_party_plot[df_party_plot['party'] == 'R']['excess_return_pct'].dropna()],
                       positions=[1, 2], showmeans=True, showmedians=True)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Democrats', 'Republicans'])
ax3.set_ylabel('Excess Return')
ax3.set_title('Return Distribution by Party')
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# 2d: Returns by chamber and party
ax4 = axes[1, 1]
chamber_party = df_party.groupby(['chamber', 'party'])['excess_return_pct'].mean().unstack()
chamber_party.plot(kind='bar', ax=ax4, color=[colors['D'], colors['R']], edgecolor='black')
ax4.set_xlabel('Chamber')
ax4.set_ylabel('Mean Excess Return')
ax4.set_title('Mean Excess Return by Chamber & Party')
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax4.tick_params(axis='x', rotation=0)
ax4.legend(title='Party')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}2_party_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}2_party_comparison.png")

# Statistical test: Party comparison
d_returns = df_party[df_party['party'] == 'D']['excess_return_pct'].dropna()
r_returns = df_party[df_party['party'] == 'R']['excess_return_pct'].dropna()
t_stat, p_value = stats.ttest_ind(d_returns, r_returns)
print(f"    T-test (D vs R): t={t_stat:.3f}, p={p_value:.4f}")

# ============================================================================
# VISUALIZATION 3: Committee Alignment Impact
# ============================================================================
print("[Creating Viz 3: Committee Alignment Analysis...]")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 3a: Aligned vs Non-aligned returns
ax1 = axes[0]
aligned_data = [df[df['aligned_trade'] == 0]['excess_return_pct'].dropna(),
                df[df['aligned_trade'] == 1]['excess_return_pct'].dropna()]
bp = ax1.boxplot(aligned_data, labels=['Non-Aligned', 'Aligned to Committee'], patch_artist=True)
bp['boxes'][0].set_facecolor('#ff6b6b')
bp['boxes'][1].set_facecolor('#4ecdc4')
ax1.set_ylabel('Excess Return')
ax1.set_title('Returns: Aligned vs Non-Aligned Trades')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# 3b: Beat market rate by alignment
ax2 = axes[1]
align_beat = df.groupby('aligned_trade')['beat_market'].mean() * 100
bars = ax2.bar(['Non-Aligned', 'Aligned'], align_beat.values, 
               color=['#ff6b6b', '#4ecdc4'], edgecolor='black')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50% (Random)')
ax2.set_ylabel('Beat Market Rate (%)')
ax2.set_title('Beat Market Rate by Committee Alignment')
ax2.legend()
for i, rate in enumerate(align_beat.values):
    ax2.annotate(f'{rate:.1f}%', xy=(i, rate), ha='center', va='bottom', fontsize=11)

# 3c: Sample size comparison
ax3 = axes[2]
align_counts = df['aligned_trade'].value_counts()
ax3.pie([align_counts.get(0, 0), align_counts.get(1, 0)], 
        labels=['Non-Aligned', 'Aligned'], autopct='%1.1f%%',
        colors=['#ff6b6b', '#4ecdc4'], startangle=90)
ax3.set_title('Trade Distribution by Alignment')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}3_committee_alignment.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}3_committee_alignment.png")

# Statistical test: Alignment
aligned_returns = df[df['aligned_trade'] == 1]['excess_return_pct'].dropna()
non_aligned_returns = df[df['aligned_trade'] == 0]['excess_return_pct'].dropna()
if len(aligned_returns) > 0 and len(non_aligned_returns) > 0:
    t_stat, p_value = stats.ttest_ind(aligned_returns, non_aligned_returns)
    print(f"    T-test (Aligned vs Non): t={t_stat:.3f}, p={p_value:.4f}")

# ============================================================================
# VISUALIZATION 4: Sector Analysis
# ============================================================================
print("[Creating Viz 4: Sector Analysis...]")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Filter out unknown sectors
df_sector = df[df['sector'] != 'Unknown'].copy()

# 4a: Number of trades by sector
ax1 = axes[0, 0]
sector_counts = df_sector['sector'].value_counts().head(10)
bars = ax1.barh(sector_counts.index, sector_counts.values, color='steelblue', edgecolor='black')
ax1.set_xlabel('Number of Trades')
ax1.set_title('Top 10 Sectors by Trade Count')
ax1.invert_yaxis()

# 4b: Mean excess return by sector
ax2 = axes[0, 1]
sector_returns = df_sector.groupby('sector')['excess_return_pct'].mean().sort_values()
sector_returns = sector_returns[sector_returns.index != 'Unknown']
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in sector_returns.values]
ax2.barh(sector_returns.index, sector_returns.values, color=colors, edgecolor='black')
ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel('Mean Excess Return')
ax2.set_title('Mean Excess Return by Sector')

# 4c: Beat market rate by sector
ax3 = axes[1, 0]
sector_beat = df_sector.groupby('sector')['beat_market'].mean() * 100
sector_beat = sector_beat.sort_values(ascending=False).head(10)
colors = ['#2ecc71' if x > 50 else '#e74c3c' for x in sector_beat.values]
ax3.barh(sector_beat.index, sector_beat.values, color=colors, edgecolor='black')
ax3.axvline(x=50, color='red', linestyle='--', linewidth=1.5)
ax3.set_xlabel('Beat Market Rate (%)')
ax3.set_title('Beat Market Rate by Sector (Top 10)')
ax3.invert_yaxis()

# 4d: Heatmap of sector vs party
ax4 = axes[1, 1]
sector_party = df_sector.pivot_table(values='excess_return_pct', 
                                     index='sector', columns='party', aggfunc='mean')
# Select top sectors
top_sectors = sector_counts.head(8).index
sector_party_top = sector_party.loc[sector_party.index.isin(top_sectors)]
sns.heatmap(sector_party_top, annot=True, fmt='.2%', cmap='RdYlGn', center=0,
             ax=ax4, cbar_kws={'label': 'Mean Excess Return'})
ax4.set_title('Excess Return: Sector x Party')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}4_sector_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}4_sector_analysis.png")

# ============================================================================
# VISUALIZATION 5: Trade Size/Value Analysis
# ============================================================================
print("[Creating Viz 5: Trade Size Analysis...]")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 5a: Trades by value bucket
ax1 = axes[0]
value_order = ['$1,001 - $15,000', '$15,001 - $50,000', '$50,001 - $100,000',
               '$100,001 - $250,000', '$250,001 - $500,000', '$500,001 - $1,000,000',
               '$1,000,001 - $5,000,000', '$5,000,001 - $25,000,000']
value_counts = df['trade_value_bucket'].value_counts()
value_counts = value_counts.reindex([v for v in value_order if v in value_counts.index])
ax1.bar(range(len(value_counts)), value_counts.values, color='steelblue', edgecolor='black')
ax1.set_xticks(range(len(value_counts)))
ax1.set_xticklabels([v.replace('$', '').replace(',001', 'K').replace(',000', 'K').replace(' - ', '-') 
                     for v in value_counts.index], rotation=45, ha='right')
ax1.set_xlabel('Trade Value Bucket')
ax1.set_ylabel('Number of Trades')
ax1.set_title('Trade Distribution by Size')

# 5b: Returns by trade size
ax2 = axes[1]
value_returns = df.groupby('trade_value_bucket')['excess_return_pct'].mean()
value_returns = value_returns.reindex([v for v in value_order if v in value_returns.index])
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in value_returns.values]
ax2.bar(range(len(value_returns)), value_returns.values * 100, color=colors, edgecolor='black')
ax2.set_xticks(range(len(value_returns)))
ax2.set_xticklabels([v.replace('$', '').replace(',001', 'K').replace(',000', 'K').replace(' - ', '-') 
                     for v in value_returns.index], rotation=45, ha='right')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel('Trade Value Bucket')
ax2.set_ylabel('Mean Excess Return (%)')
ax2.set_title('Mean Returns by Trade Size')

# 5c: Trade size vs beat market rate
ax3 = axes[2]
value_beat = df.groupby('trade_value_bucket')['beat_market'].mean() * 100
value_beat = value_beat.reindex([v for v in value_order if v in value_beat.index])
ax3.bar(range(len(value_beat)), value_beat.values, color='steelblue', edgecolor='black')
ax3.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50% (Random)')
ax3.set_xticks(range(len(value_beat)))
ax3.set_xticklabels([v.replace('$', '').replace(',001', 'K').replace(',000', 'K').replace(' - ', '-') 
                     for v in value_beat.index], rotation=45, ha='right')
ax3.set_xlabel('Trade Value Bucket')
ax3.set_ylabel('Beat Market Rate (%)')
ax3.set_title('Beat Market Rate by Trade Size')
ax3.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}5_trade_size.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}5_trade_size.png")

# ============================================================================
# VISUALIZATION 6: Disclosure Lag Analysis
# ============================================================================
print("[Creating Viz 6: Disclosure Lag Analysis...]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6a: Distribution of disclosure lag
ax1 = axes[0, 0]
lag_data = df['disclosure_lag_days'].clip(upper=365)  # Cap at 1 year for visualization
ax1.hist(lag_data, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax1.axvline(x=30, color='blue', linestyle='--', linewidth=2, label='30 days')
ax1.axvline(x=lag_data.median(), color='green', linestyle='-', linewidth=2, 
            label=f'Median: {lag_data.median():.0f} days')
ax1.set_xlabel('Disclosure Lag (days, capped at 365)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Disclosure Lags')
ax1.legend()

# 6b: Disclosure lag over time
ax2 = axes[0, 1]
df['year'] = df['trade_date'].dt.year
lag_by_year = df.groupby('year')['disclosure_lag_days'].median()
ax2.plot(lag_by_year.index, lag_by_year.values, marker='o', linewidth=2, color='steelblue')
ax2.fill_between(lag_by_year.index, lag_by_year.values, alpha=0.3)
ax2.set_xlabel('Year')
ax2.set_ylabel('Median Disclosure Lag (days)')
ax2.set_title('Median Disclosure Lag Over Time')

# 6c: Lag vs returns
ax3 = axes[1, 0]
sample = df.sample(min(5000, len(df)), random_state=42)
ax3.scatter(sample['disclosure_lag_days'], sample['excess_return_pct'], 
            alpha=0.3, s=10, c='steelblue')
ax3.set_xlabel('Disclosure Lag (days)')
ax3.set_ylabel('Excess Return')
ax3.set_title('Disclosure Lag vs Returns')
ax3.set_xlim(0, 365)

# 6d: Beat market rate by lag quartiles
ax4 = axes[1, 1]
df['lag_quartile'] = pd.qcut(df['disclosure_lag_days'], q=4, labels=['Q1 (Fast)', 'Q2', 'Q3', 'Q4 (Slow)'])
lag_beat = df.groupby('lag_quartile')['beat_market'].mean() * 100
ax4.bar(lag_beat.index.astype(str), lag_beat.values, color='steelblue', edgecolor='black')
ax4.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50%')
ax4.set_xlabel('Disclosure Lag Quartile')
ax4.set_ylabel('Beat Market Rate (%)')
ax4.set_title('Beat Market Rate by Disclosure Speed')
ax4.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}6_disclosure_lag.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}6_disclosure_lag.png")

# ============================================================================
# VISUALIZATION 7: Temporal Trends
# ============================================================================
print("[Creating Viz 7: Temporal Trends...]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7a: Trades over time
ax1 = axes[0, 0]
trades_by_year = df.groupby('year').size()
ax1.bar(trades_by_year.index, trades_by_year.values, color='steelblue', edgecolor='black')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Trades')
ax1.set_title('Congressional Trades by Year')

# 7b: Returns over time (quarterly)
ax7 = axes[0, 1]
df['quarter'] = df['trade_date'].dt.to_period('Q')
returns_by_quarter = df.groupby('quarter')['excess_return_pct'].mean()
returns_by_quarter.index = returns_by_quarter.index.to_timestamp()
ax7.plot(returns_by_quarter.index, returns_by_quarter.values * 100, linewidth=2, color='steelblue')
ax7.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax7.fill_between(returns_by_quarter.index, returns_by_quarter.values * 100, alpha=0.3)
ax7.set_xlabel('Quarter')
ax7.set_ylabel('Mean Excess Return (%)')
ax7.set_title('Quarterly Mean Excess Returns')

# 7c: Party composition over time
ax3 = axes[1, 0]
party_by_year = df[df['party'].isin(['D', 'R'])].groupby(['year', 'party']).size().unstack()
party_colors = {'D': '#3182bd', 'R': '#e6550d'}
party_by_year.plot(kind='bar', stacked=True, ax=ax3, color=[party_colors['D'], party_colors['R']], edgecolor='black')
ax3.set_xlabel('Year')
ax3.set_ylabel('Number of Trades')
ax3.set_title('Trade Volume by Party Over Time')
ax3.legend(title='Party')
ax3.tick_params(axis='x', rotation=45)

# 7d: Beat market rate over time
ax4 = axes[1, 1]
bm_by_year = df.groupby('year')['beat_market'].mean() * 100
ax4.plot(bm_by_year.index, bm_by_year.values, marker='o', linewidth=2, color='steelblue')
ax4.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50% (Random)')
ax4.set_xlabel('Year')
ax4.set_ylabel('Beat Market Rate (%)')
ax4.set_title('Beat Market Rate by Year')
ax4.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}7_temporal_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}7_temporal_trends.png")

# ============================================================================
# VISUALIZATION 8: Top Traders Analysis
# ============================================================================
print("[Creating Viz 8: Top Traders Analysis...]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 8a: Top 10 most active traders
ax1 = axes[0, 0]
top_traders = df['member_name'].value_counts().head(10)
ax1.barh(top_traders.index, top_traders.values, color='steelblue', edgecolor='black')
ax1.set_xlabel('Number of Trades')
ax1.set_title('Top 10 Most Active Traders')
ax1.invert_yaxis()

# 8b: Top performers by mean return
ax2 = axes[0, 1]
min_trades = 20  # Minimum trades for inclusion
trader_returns = df.groupby('member_name').agg({
    'excess_return_pct': 'mean',
    'beat_market': 'mean',
    'ticker': 'count'
}).rename(columns={'ticker': 'trade_count'})
trader_returns = trader_returns[trader_returns['trade_count'] >= min_trades]
top_performers = trader_returns.nlargest(10, 'excess_return_pct')
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_performers['excess_return_pct']]
ax2.barh(top_performers.index, top_performers['excess_return_pct'].values * 100, 
         color=colors, edgecolor='black')
ax2.set_xlabel('Mean Excess Return (%)')
ax2.set_title(f'Top 10 Performers (min {min_trades} trades)')
ax2.invert_yaxis()

# 8c: Bottom performers
ax3 = axes[1, 0]
bottom_performers = trader_returns.nsmallest(10, 'excess_return_pct')
colors = ['#e74c3c' for x in bottom_performers['excess_return_pct']]
ax3.barh(bottom_performers.index, bottom_performers['excess_return_pct'].values * 100, 
         color=colors, edgecolor='black')
ax3.set_xlabel('Mean Excess Return (%)')
ax3.set_title(f'Bottom 10 Performers (min {min_trades} trades)')
ax3.invert_yaxis()

# 8d: Beat market rate for top traders
ax4 = axes[1, 1]
top_bm = trader_returns.nlargest(10, 'beat_market')
ax4.barh(top_bm.index, top_bm['beat_market'].values * 100, color='steelblue', edgecolor='black')
ax4.axvline(x=50, color='red', linestyle='--', linewidth=1.5)
ax4.set_xlabel('Beat Market Rate (%)')
ax4.set_title(f'Top 10 Beat-Market Rate (min {min_trades} trades)')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}8_top_traders.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}8_top_traders.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# Key findings
print("\nKEY FINDINGS:")
print("-" * 50)

# Overall performance
overall_beat = df['beat_market'].mean() * 100
mean_return = df['excess_return_pct'].mean() * 100
print(f"1. OVERALL PERFORMANCE:")
print(f"   - Beat Market Rate: {overall_beat:.1f}%")
print(f"   - Mean Excess Return: {mean_return:.2f}%")
print(f"   - Interpretation: Congressional trades UNDERPERFORM the market on average")

# Party comparison
print(f"\n2. PARTY COMPARISON:")
for party in ['D', 'R']:
    party_data = df[df['party'] == party]
    party_beat = party_data['beat_market'].mean() * 100
    party_return = party_data['excess_return_pct'].mean() * 100
    print(f"   - {'Democrats' if party == 'D' else 'Republicans'}: Beat {party_beat:.1f}%, Mean Return {party_return:.2f}%")

# Committee alignment
print(f"\n3. COMMITTEE ALIGNMENT:")
aligned_beat = df[df['aligned_trade'] == 1]['beat_market'].mean() * 100
non_aligned_beat = df[df['aligned_trade'] == 0]['beat_market'].mean() * 100
print(f"   - Aligned Trades: Beat {aligned_beat:.1f}%")
print(f"   - Non-Aligned Trades: Beat {non_aligned_beat:.1f}%")

# Trade size
print(f"\n4. TRADE SIZE ANALYSIS:")
small_trades = df[df['trade_value_bucket'].isin(['$1,001 - $15,000', '$15,001 - $50,000'])]
large_trades = df[df['trade_value_bucket'].isin(['$500,001 - $1,000,000', '$1,000,001 - $5,000,000'])]
print(f"   - Small trades (<$50K): Beat {small_trades['beat_market'].mean()*100:.1f}%")
print(f"   - Large trades (>$500K): Beat {large_trades['beat_market'].mean()*100:.1f}%")

# Disclosure
print(f"\n5. DISCLOSURE LAG:")
print(f"   - Median: {df['disclosure_lag_days'].median():.0f} days")
print(f"   - Mean: {df['disclosure_lag_days'].mean():.0f} days")

# Statistical tests summary
print(f"\nSTATISTICAL TESTS:")
print(f"   - One-sample t-test (H0: mean return = 0)")
t_stat, p_val = stats.ttest_1samp(df['excess_return_pct'].dropna(), 0)
print(f"     t={t_stat:.3f}, p={p_val:.4f} {'(Significant!)' if p_val < 0.05 else '(Not significant)'}")

# Save summary to file
with open(f'{OUTPUT_DIR}analysis_summary.txt', 'w') as f:
    f.write("CONGRESSIONAL TRADING EDA - KEY FINDINGS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset: {len(df):,} trades from {df['member_name'].nunique()} members\n")
    f.write(f"Date range: {df['trade_date'].min().date()} to {df['trade_date'].max().date()}\n\n")
    f.write(f"Beat Market Rate: {overall_beat:.1f}%\n")
    f.write(f"Mean Excess Return: {mean_return:.2f}%\n\n")
    f.write("This suggests that on average, Congressional trades UNDERPERFORM the S&P 500.\n")

print(f"\nAnalysis complete! Output saved to: {OUTPUT_DIR}")
print(f"   Generated 8 visualization files")
