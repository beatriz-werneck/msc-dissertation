# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_36_individual_equity — individual-level index by economic class
# -------------------------------------------------------------------------
"""
Moves the analysis from the origin point to the PERSON. The index is computed per
origin, but a person usually has several origins, so three measures are derived
for each individual (id_pess):
  - total       : mean index over all the person's anchor origins
  - residential : index at the home origin (motivo_o = 8)
  - work/study  : index at the work or study origin (motivo_o in 1..4)
These are then compared across Critério Brasil economic classes (A ... D-E),
expansion-weighted by the person factor fe_pess so the results represent the
population. The work - home gap per class is also reported.

Headline threshold: 15 min (20 and 30 min class means printed for reference).
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bcw_dis_00_config as cfg

CLASS_ORDER = ["A", "B1", "B2", "C1", "C2", "D-E"]
HOME, WORK = {8}, {1, 2, 3, 4}
THRESHOLDS = [15, 20, 30]
BAR_T = 15
BARC = {"Total": plt.cm.viridis(0.20), "Residential": plt.cm.viridis(0.55),
        "Work/Study": plt.cm.viridis(0.85)}

df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index_rich.parquet")
df["motivo_o"] = df["motivo_o"].astype("Int64")


def wmean(s, wt):
    m = s.notna() & wt.notna()
    return np.average(s[m], weights=wt[m]) if m.any() else np.nan


# -------------------------------------------------------------------------
# 1. Per-person measures (total / residential / work) at each threshold
# -------------------------------------------------------------------------
per = df.groupby("id_pess").agg(classe=("classe_economica", "first"),
                                fe=("fe_pess", "first"))
for T in THRESHOLDS:
    idx = f"index_coverage_{T}"
    per[f"total_{T}"] = df.groupby("id_pess")[idx].mean()
    per[f"resid_{T}"] = df[df["motivo_o"].isin(HOME)].groupby("id_pess")[idx].mean()
    per[f"work_{T}"] = df[df["motivo_o"].isin(WORK)].groupby("id_pess")[idx].mean()
per["gap_15"] = per["work_15"] - per["resid_15"]        # work - home
per = per.reset_index()
per.to_parquet(cfg.OUTPUT_DIR / "individual_index.parquet", index=False)
print(f"persons: {len(per):,}")

# -------------------------------------------------------------------------
# 2. Expansion-weighted class means
# -------------------------------------------------------------------------
def class_table(T):
    rows = []
    for cls in CLASS_ORDER:
        s = per[per["classe"] == cls]
        rows.append([cls,
                     wmean(s[f"total_{T}"], s["fe"]),
                     wmean(s[f"resid_{T}"], s["fe"]),
                     wmean(s[f"work_{T}"], s["fe"]),
                     wmean(s[f"work_{T}"] - s[f"resid_{T}"], s["fe"]),
                     len(s)])
    return pd.DataFrame(rows, columns=["Class", "Total", "Residential",
                                       "Work/Study", "Work-Home gap", "Persons"])


tab15 = class_table(15)
print("\n=== 15-min index by economic class (expansion-weighted) ===")
print(tab15.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
for T in (20, 30):
    t = class_table(T)
    print(f"\n--- {T} min (Total / Residential / Work) ---")
    print(t[["Class", "Total", "Residential", "Work/Study"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
tab15.to_csv(cfg.OUTPUT_DIR / "individual_equity_15min.csv", index=False)

# -------------------------------------------------------------------------
# 3. Grouped bar chart (15 min): Total / Residential / Work by class
# -------------------------------------------------------------------------
measures = ["Total", "Residential", "Work/Study"]
x = np.arange(len(CLASS_ORDER)); wbar = 0.26
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.patch.set_alpha(0)
for i, m in enumerate(measures):
    ax.bar(x + (i - 1) * wbar, tab15[m].values, wbar, label=m,
           color=BARC[m], edgecolor="white", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(CLASS_ORDER)
ax.set_xlabel("Economic class", fontsize=10)
ax.set_ylabel("X-Minute City Accessibility Index (15 min)", fontsize=10)
ax.set_ylim(0, max(tab15[measures].max()) * 1.15)
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
ax.grid(axis="y", alpha=0.3)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "36_individual_equity_bars_15min.png"
fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
print(f"\nSaved figure: {out}")

# -------------------------------------------------------------------------
# 4. Class-means table (PNG)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 2.9)); ax.axis("off")
disp = tab15.copy()
for c in ["Total", "Residential", "Work/Study", "Work-Home gap"]:
    disp[c] = disp[c].map(lambda v: f"{v:.3f}")
t = ax.table(cellText=disp.values, colLabels=disp.columns, cellLoc="center", loc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
for (r, cc), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor("black"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f0f0f0")
out = cfg.FIG_DIR / "36_individual_equity_table_15min.png"
fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
print(f"Saved table: {out}")
