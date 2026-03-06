#!/usr/bin/env python3
"""
visualize_regions.py

Produces two region maps from regions_computed.csv + uszips.csv (simplemaps zip):
  1. Inline Plotly figure (for use in Jupyter notebook)
  2. region_map.html — MA + neighbors, saved for sharing/presentation

Run generate_regions.py first to produce regions_computed.csv.
"""

import zipfile
import pandas as pd
import plotly.express as px

INPUT_ZIP  = 'simplemaps_uszips_basicv1.94.zip'
REGIONS_CSV = 'regions_computed.csv'
OUTPUT_HTML = 'region_map.html'

# States to include in the HTML map (MA + neighbors worth showing)
NEIGHBOR_STATES = {'MA', 'CT', 'RI', 'NH', 'VT', 'ME', 'NY', 'NJ', 'PA'}

# Consistent color palette — one color per region, keyed by name
REGION_COLORS = {
    'Wor Central':  '#1a237e',   # deep navy
    'Wor North':    '#1565c0',   # dark blue
    'Wor West':     '#1976d2',   # medium blue
    'Wor South':    '#42a5f5',   # light blue
    'MW/495':       '#00897b',   # teal
    'Boston Reg':   '#e53935',   # red
    'North Shore':  '#f57c00',   # orange
    'RI/Cape':      '#8e24aa',   # purple
    'South':        '#ad1457',   # deep pink (CT only)
    'West MA':      '#6d4c41',   # brown
    'North':        '#78909c',   # slate
    'NY Area':      '#43a047',   # green
    'Far Away':     '#bdbdbd',   # light gray
}


def load_data():
    # Load ZIP centroids from simplemaps zip
    with zipfile.ZipFile(INPUT_ZIP) as z:
        zips = pd.read_csv(z.open('uszips.csv'), dtype={'zip': str},
                           usecols=['zip', 'lat', 'lng', 'city', 'state_id'],
                           low_memory=False)

    # Load computed regions
    regions = pd.read_csv(REGIONS_CSV, dtype={'ZIP': str},
                          usecols=['ZIP', 'RegionAssignment'])
    regions['ZIP'] = regions['ZIP'].str.zfill(5)

    df = zips.merge(regions, left_on='zip', right_on='ZIP', how='left')
    df['RegionAssignment'] = df['RegionAssignment'].fillna('Far Away')
    return df


def make_map(df, title, output_html=None):
    """Build a Plotly scatter_map figure. Saves HTML if output_html is given."""
    # Sort so legend appears in a logical order
    region_order = list(REGION_COLORS.keys())
    df = df.copy()
    df['_sort'] = df['RegionAssignment'].map(
        {r: i for i, r in enumerate(region_order)}
    ).fillna(len(region_order))
    df = df.sort_values('_sort')

    colors_present = [r for r in region_order if r in df['RegionAssignment'].values]

    fig = px.scatter_map(
        df,
        lat='lat', lon='lng',
        color='RegionAssignment',
        color_discrete_map=REGION_COLORS,
        category_orders={'RegionAssignment': colors_present},
        hover_name='city',
        hover_data={'state_id': True, 'RegionAssignment': True,
                    'lat': False, 'lng': False, '_sort': False},
        zoom=6,
        title=title,
        map_style='carto-positron',
        opacity=0.7,
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(
        legend_title_text='Region',
        margin={'r': 0, 't': 40, 'l': 0, 'b': 0},
        height=700,
    )

    if output_html:
        fig.write_html(output_html)
        print(f'Saved {output_html}')

    return fig


def plot_notebook(full=False):
    """
    Call from a notebook cell.
    full=False → MA + neighbors only (default, faster)
    full=True  → all US ZIPs
    """
    df = load_data()
    if not full:
        df = df[df['state_id'].isin(NEIGHBOR_STATES)]
        title = 'Music Worcester Patron Regions — MA & Neighbors'
    else:
        title = 'Music Worcester Patron Regions — Full US'
    return make_map(df, title)


def main():
    df = load_data()

    # HTML: MA + neighbors
    neighbors = df[df['state_id'].isin(NEIGHBOR_STATES)].copy()
    make_map(neighbors,
             'Music Worcester Patron Regions — MA & Neighbors',
             output_html=OUTPUT_HTML)
    print(f'Done. Open {OUTPUT_HTML} in a browser to view.')


if __name__ == '__main__':
    main()
