"""
Portfolio visualizations: Forecast vs Actuals — anonymized.
No event names, venue names, or organization identifiers.

Output files:
  forecast_portfolio_scatter.png    (3-panel scatter by season)
  forecast_portfolio_accuracy.png   (MAPE + bias by season, incl. 25-26 partial)
  forecast_portfolio_class.png      (MAPE + bias by event type, ex-Prestige)
  forecast_portfolio_combined.png   (combined layout)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

WORKING_DIR = '/Users/antho/Documents/WPI-MW'
os.chdir(WORKING_DIR)

# ── Palette ────────────────────────────────────────────────────────────────
NAVY   = '#1A3A5C'
ORANGE = '#E8922A'
TEAL   = '#2A9EA0'
GOLD   = '#C4A35A'
LGRAY  = '#E8EDF2'
DGRAY  = '#5A6A7A'
PGRAY  = '#AABBCC'   # partial bar color

CLASS_COLORS = {
    'Headliner': NAVY,
    'Standard':  TEAL,
    'Mission':   ORANGE,
    'Prestige':  GOLD,
}

SEASON_LABELS = {'22-23': "'22–23", '23-24': "'23–24", '24-25': "'24–25"}

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.labelcolor':   DGRAY,
    'xtick.color':       DGRAY,
    'ytick.color':       DGRAY,
    'text.color':        NAVY,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
})


# ── Data loading ───────────────────────────────────────────────────────────
def load():
    df = pd.read_excel('Forecast_Honest_Eval_Patched.xlsx', sheet_name='All_Seasons')
    df = df[df['ActualAttendance'] > 0].copy()
    df['AbsPct']    = df['PercentError'] * 100
    df['SignedPct'] = df['SignedPctError'] * 100
    df['SeasonLabel'] = df['Season'].map(SEASON_LABELS).fillna(df['Season'])
    return df


def load_2526_partial():
    """12 events with reliable actuals (EventDate ≤ Dec 11 2025).
    Matches the 2024-12-11 SF export cutoff; post-Dec-11 rows show only
    advance ticket sales as 'actuals' and are excluded."""
    comp = pd.read_excel('Forecast_2526_Comparison.xlsx', sheet_name='2526_Comparison')
    em   = pd.read_excel('EventManifest.xlsx', sheet_name='EventManifest')

    dates = em.drop_duplicates('EventName')[['EventName', 'EventDate']].copy()
    dates['EventDate'] = pd.to_datetime(dates['EventDate'], errors='coerce')

    comp = comp.merge(dates, on='EventName', how='left')
    cutoff = pd.Timestamp('2025-12-11')
    rel = comp[comp['EventDate'] <= cutoff].copy()

    rel['AbsPct']    = (rel['Pred_Adj'] - rel['Actual']).abs() / rel['Actual'] * 100
    rel['SignedPct'] = (rel['Pred_Adj'] - rel['Actual'])       / rel['Actual'] * 100

    return {
        'season':  "'25–26*",
        'MAPE':    rel['AbsPct'].mean(),
        'Bias':    rel['SignedPct'].mean(),
        'n':       len(rel),
        'partial': True,
    }


# ── Metrics helpers ────────────────────────────────────────────────────────
def season_metrics(df, row_2526=None):
    rows = []
    for season, g in df.groupby('Season'):
        rows.append({
            'season':  SEASON_LABELS[season],
            'MAPE':    g['AbsPct'].mean(),
            'Bias':    g['SignedPct'].mean(),
            'n':       len(g),
            'partial': False,
        })
    if row_2526:
        rows.append(row_2526)
    return pd.DataFrame(rows)


def class_metrics(df):
    rows = []
    for cls, g in df.groupby('EventClass'):
        rows.append({
            'Class': cls,
            'MAPE':  g['AbsPct'].mean(),
            'Bias':  g['SignedPct'].mean(),
            'n':     len(g),
        })
    return pd.DataFrame(rows).sort_values('MAPE')


# ── Axis styling ───────────────────────────────────────────────────────────
def style_axis(ax, title=None, xlabel=None, ylabel=None, pct_y=False, pct_x=False):
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color=NAVY, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=DGRAY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=DGRAY)
    ax.tick_params(labelsize=8.5)
    ax.grid(axis='y', color=LGRAY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_color(LGRAY)
    ax.spines['bottom'].set_color(LGRAY)
    if pct_y:
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    if pct_x:
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))


# ── Chart 1: 3-panel scatter by season ────────────────────────────────────
def plot_scatter_grid(df, axes):
    seasons = sorted(df['Season'].unique())
    global_max = max(df['ActualAttendance'].max(), df['PredictedAttendance'].max()) * 1.08

    for ax, season in zip(axes, seasons):
        g = df[df['Season'] == season]
        lim = global_max
        ax.plot([0, lim], [0, lim], color=DGRAY, linewidth=0.9,
                linestyle='--', zorder=1)
        for cls, sub in g.groupby('EventClass'):
            color = CLASS_COLORS.get(cls, DGRAY)
            ax.scatter(sub['PredictedAttendance'], sub['ActualAttendance'],
                       color=color, s=52, alpha=0.85,
                       edgecolors='white', linewidths=0.4,
                       label=cls, zorder=3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal', adjustable='box')

        mape = g['AbsPct'].mean()
        bias = g['SignedPct'].mean()
        ax.text(0.97, 0.05,
                f'MAPE {mape:.0f}%\nBias {bias:+.0f}%\nn={len(g)}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color=DGRAY,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=LGRAY, alpha=0.9))

        label = SEASON_LABELS.get(season, season)
        style_axis(ax, title=label,
                   xlabel='Predicted tickets', ylabel='Actual tickets')

        if ax == axes[0]:
            legend = ax.legend(fontsize=8, frameon=False, loc='upper left',
                               handletextpad=0.3, labelspacing=0.3,
                               markerscale=0.8)


# ── Chart 2: MAPE by season + bias secondary axis ─────────────────────────
def plot_season_accuracy(sm, ax):
    x = np.arange(len(sm))
    bar_colors = [PGRAY if row.get('partial') else NAVY for _, row in sm.iterrows()]
    bars = ax.bar(x, sm['MAPE'], color=bar_colors, width=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(sm['season'], fontsize=9.5)
    ax.set_ylim(0, sm['MAPE'].max() * 1.5)

    for bar, mape, partial in zip(bars, sm['MAPE'], sm.get('partial', [False]*len(sm))):
        color = DGRAY if partial else NAVY
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{mape:.0f}%', ha='center', va='bottom',
                fontsize=8.5, color=color, fontweight='bold')

    # Bias — secondary axis
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(LGRAY)
    bias_vals = sm['Bias'].values
    ax2.plot(x, bias_vals, color=ORANGE, linewidth=1.8, linestyle='-',
             marker='o', markersize=7, zorder=5, label='Bias')
    ax2.axhline(0, color=ORANGE, linewidth=0.8, linestyle='--', alpha=0.45, zorder=4)
    for xi, bv, partial in zip(x, bias_vals, sm.get('partial', [False]*len(sm))):
        offset = 2.5 if bv >= 0 else -5
        ax2.text(xi + 0.17, bv + offset, f'{bv:+.0f}%',
                 fontsize=8, color=ORANGE, fontweight='bold',
                 alpha=0.7 if partial else 1.0)
    bias_lim = max(abs(bias_vals.min()), abs(bias_vals.max())) * 3.5
    ax2.set_ylim(-bias_lim, bias_lim)
    ax2.set_ylabel('Bias %  (+ = over-forecast)', fontsize=8.5, color=ORANGE)
    ax2.tick_params(axis='y', colors=ORANGE, labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    # Partial bar annotation
    ax.text(0.99, 0.02, '* Pre-Dec 2025 events only',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.5, color=DGRAY, style='italic')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, ['MAPE'] + l2, fontsize=8.5, frameon=False,
              loc='upper left', handletextpad=0.4)

    ax.text(0.01, 0.96, f'n = {sm["n"].sum()} events',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8, color=DGRAY)
    style_axis(ax, title='Forecast Accuracy by Season',
               ylabel='MAPE (%)', pct_y=True)


# ── Chart 3: MAPE + bias by event type (ex-Prestige) ──────────────────────
def plot_class_mape(cm, ax):
    colors = [CLASS_COLORS.get(c, DGRAY) for c in cm['Class']]
    y = np.arange(len(cm))
    bars = ax.barh(y, cm['MAPE'], color=colors, height=0.55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(cm['Class'], fontsize=10)
    ax.set_xlim(0, cm['MAPE'].max() * 1.5)
    for bar, mape, bias, n in zip(bars, cm['MAPE'], cm['Bias'], cm['n']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{mape:.0f}%  bias {bias:+.0f}%  (n={n})',
                va='center', fontsize=8.5, color=NAVY)
    ax.grid(axis='x', color=LGRAY, linewidth=0.7, zorder=0)
    ax.grid(axis='y', visible=False)
    style_axis(ax, title='Forecast Error by Event Type', xlabel='MAPE (%)', pct_x=True)


# ── Save individual PNGs ───────────────────────────────────────────────────
def save_individual(df, sm, cm):
    # 3-panel scatter
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    plot_scatter_grid(df, axes)
    fig.suptitle('Predicted vs Actual Attendance by Season',
                 fontsize=12, fontweight='bold', color=NAVY, y=1.02)
    fig.tight_layout()
    fig.savefig('forecast_portfolio_scatter.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ forecast_portfolio_scatter.png")

    # Season accuracy
    fig, ax = plt.subplots(figsize=(6.5, 4))
    plot_season_accuracy(sm, ax)
    fig.tight_layout()
    fig.savefig('forecast_portfolio_accuracy.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ forecast_portfolio_accuracy.png")

    # Class MAPE
    fig, ax = plt.subplots(figsize=(6.5, 3))
    plot_class_mape(cm, ax)
    fig.tight_layout()
    fig.savefig('forecast_portfolio_class.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ forecast_portfolio_class.png")


# ── Combined layout ────────────────────────────────────────────────────────
def save_combined(df, sm, cm):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Top row: 3-panel scatter
    scatter_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    plot_scatter_grid(df, scatter_axes)

    # Bottom left (spans 2 cols): season accuracy
    ax_acc = fig.add_subplot(gs[1, :2])
    plot_season_accuracy(sm, ax_acc)

    # Bottom right: class MAPE
    ax_cls = fig.add_subplot(gs[1, 2])
    plot_class_mape(cm, ax_cls)

    fig.suptitle('Attendance Forecasting Model — Evaluation Results',
                 fontsize=14, fontweight='bold', color=NAVY, y=1.01)
    fig.text(0.5, -0.01,
             'Temporal holdout evaluation  ·  3 seasons  ·  80 events  ·  Anonymized',
             ha='center', fontsize=8.5, color=DGRAY)

    fig.savefig('forecast_portfolio_combined.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ forecast_portfolio_combined.png")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    df      = load()
    row2526 = load_2526_partial()
    sm      = season_metrics(df, row_2526=row2526)
    cm      = class_metrics(df)

    print(f"Loaded {len(df)} events (3 seasons)")
    print(f"25-26 partial: n={row2526['n']}  MAPE={row2526['MAPE']:.1f}%  Bias={row2526['Bias']:+.1f}%")
    print("\nSeason metrics:")
    print(sm.to_string(index=False))
    print("\nClass metrics (ex-Prestige):")
    print(cm.to_string(index=False))

    print("\nGenerating charts...")
    save_individual(df, sm, cm)
    save_combined(df, sm, cm)
    print("\nDone.")


if __name__ == '__main__':
    main()
