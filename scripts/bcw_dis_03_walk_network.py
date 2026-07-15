# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_03_walk_network — OSM pedestrian network for SP
# -------------------------------------------------------------------------
"""
Download / load and explore the São Paulo walking network (foot-accessible
ways) from OpenStreetMap via OSMnx. Cached as GraphML for reuse by the index.

Dependencies: osmnx, geopandas, matplotlib.
"""
# %%
import networkx as nx
import osmnx as ox

import bcw_dis_00_config as cfg

cfg.WALK_DIR.mkdir(parents=True, exist_ok=True)

# %%
if cfg.WALK_GRAPHML.exists():
    print("Loading cached walk network...")
    G_walk = ox.load_graphml(cfg.WALK_GRAPHML)
else:
    print("Downloading OSM walk network for São Paulo (a few minutes)...")
    G_walk = ox.graph_from_place(
        cfg.PLACE_NAME,
        network_type="walk",     # foot-accessible ways
        simplify=True,           # drop degree-2 nodes for cleaner topology
        retain_all=True,         # keep disconnected components
    )
    ox.save_graphml(G_walk, cfg.WALK_GRAPHML)
    print(f"Saved walk network to: {cfg.WALK_GRAPHML}")

nodes_walk, edges_walk = ox.graph_to_gdfs(G_walk)

# %% [markdown]
## Data exploration
# %%
print("=" * 60)
print(f"Walk network: {len(nodes_walk):,} nodes, {len(edges_walk):,} edges")
print(f"Nodes CRS: {nodes_walk.crs} | Edges CRS: {edges_walk.crs}")
print("=" * 60)

print("\n--- Edge attributes ---")
print(edges_walk.dtypes)

print("\n--- Edge length (m) summary ---")
if "length" in edges_walk.columns:
    print(edges_walk["length"].describe())

# Connectedness: how much of the network is in the largest component.
n_components = nx.number_weakly_connected_components(G_walk)
largest = max(nx.weakly_connected_components(G_walk), key=len)
print(f"\nWeakly-connected components: {n_components:,}")
print(f"Largest component holds {len(largest)/len(nodes_walk)*100:.1f}% of nodes")

print("\n--- Node sample ---")
print(nodes_walk.head())

# %%
import matplotlib.pyplot as plt

fig, ax = ox.plot_graph(
    G_walk, node_size=0, edge_linewidth=0.2, edge_color="#3366cc",
    bgcolor="white", show=False, close=False,
)
ax.set_title(f"São Paulo walk network ({len(edges_walk):,} edges)")
cfg.save_fig("03_walk_network", fig)

# %% [markdown]
# Exploration notes / suggestions:
# - `retain_all=True` keeps isolated subgraphs; the index snaps origins to the
#   nearest node, so a tiny disconnected component could trap an origin. The
#   largest-component share above tells you how big that risk is.
# - Edge `length` is in metres (graph is projected internally by OSMnx); the
#   index converts length -> minutes using walk speed + SRTM grade penalty.
# - Watch for zero/near-zero length edges (artifacts) — the index filters them.
