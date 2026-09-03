# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_04_bike_network — OSM cycling network for SP
# -------------------------------------------------------------------------
"""
Download / load and explore the São Paulo cycling network from OpenStreetMap
via OSMnx. The 'bike' filter keeps cycleways, bicycle=yes ways and roads where
cycling is permitted, excluding motorways. Cached as GraphML for the index.

Dependencies: osmnx, geopandas, matplotlib.
"""
# %%
import networkx as nx
import osmnx as ox

import bcw_dis_00_config as cfg

cfg.BIKE_DIR.mkdir(parents=True, exist_ok=True)

# %%
if cfg.BIKE_GRAPHML.exists():
    print("Loading cached bike network...")
    G_bike = ox.load_graphml(cfg.BIKE_GRAPHML)
else:
    print("Downloading OSM bike network for São Paulo (a few minutes)...")
    G_bike = ox.graph_from_place(
        cfg.PLACE_NAME,
        network_type="bike",
        simplify=True,
        retain_all=True,
    )
    ox.save_graphml(G_bike, cfg.BIKE_GRAPHML)
    print(f"Saved bike network to: {cfg.BIKE_GRAPHML}")

nodes_bike, edges_bike = ox.graph_to_gdfs(G_bike)

# %% [markdown]
## Data exploration
# %%
print("=" * 60)
print(f"Bike network: {len(nodes_bike):,} nodes, {len(edges_bike):,} edges")
print(f"Nodes CRS: {nodes_bike.crs} | Edges CRS: {edges_bike.crs}")
print("=" * 60)

print("\n--- Edge length (m) summary ---")
if "length" in edges_bike.columns:
    print(edges_bike["length"].describe())

# Highway-type composition (what kinds of ways cyclists are routed on).
if "highway" in edges_bike.columns:
    print("\n--- Edge highway types (top 10) ---")
    print(edges_bike["highway"].astype(str).value_counts().head(10))

n_components = nx.number_weakly_connected_components(G_bike)
largest = max(nx.weakly_connected_components(G_bike), key=len)
print(f"\nWeakly-connected components: {n_components:,}")
print(f"Largest component holds {len(largest)/len(nodes_bike)*100:.1f}% of nodes")

# %%
import matplotlib.pyplot as plt

fig, ax = ox.plot_graph(
    G_bike, node_size=0, edge_linewidth=0.2, edge_color="#2ca02c",
    bgcolor="white", show=False, close=False,
)
ax.set_title(f"São Paulo bike network ({len(edges_bike):,} edges)")
cfg.save_fig("04_bike_network", fig)
