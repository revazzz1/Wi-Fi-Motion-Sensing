"""Stage 1 exploratory analysis: Wi-Fi CSI presence detection (room4, empty vs one person).

Data: IEEE DataPort "CSI Dataset for WiFi Based Human Detection" (DOI 10.21227/smn7-jb67),
unzipped into the project as "CSV Files/". Each magnitude CSV = one ~70 s recording,
rows = packets (time), 49 columns = subcarriers, no header, no timestamps.

Outputs go to figures/: fig_raw_subcarrier0.png, fig_a_heatmap.png,
fig_b_spread_boxplot.png, table_dataset_stats.csv
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = "CSV Files"
FOLDERS = {
    "empty":      f"{DATA}/room4_iqbal/Static_noperson_channel_11_20mhz/csi_magnitude_data",
    "person":     f"{DATA}/room4_iqbal/Dynamic_1_person_ch11_20mhz/csi_magnitude_data",
    # not used for training: only checked to decide what "Static" means (see label check below)
    "lab_static": f"{DATA}/LAB_Final Year Project/Static_Lab_ch6_20mhz/csi_magnitude_data",
}
FILE_S = 70    # seconds between consecutive file start times (from the filenames)
WIN_S = 2      # window length in seconds
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ---------- helpers ----------

def list_csv(folder):
    return sorted(glob.glob(folder + "/*.csv"))


def load(path):
    return pd.read_csv(path, header=None)   # no header row: the first line is data


def spread(df):
    """Median over subcarriers of the per-subcarrier IQR. Robust to the rare amplitude spikes."""
    iqr = df.quantile(0.75) - df.quantile(0.25)   # 49 numbers, one per subcarrier
    return iqr.median()


def window(df, start_s, length_s=WIN_S, file_s=FILE_S):
    """Cut a length_s-second window starting at start_s, using this file's own packet rate."""
    rate = len(df) / file_s                      # estimated packets per second for this file
    a = int(start_s * rate)
    b = a + int(length_s * rate)
    return df.iloc[a:b].to_numpy()               # shape: (packets, 49)


def summarise_folder(files):
    """Load each file ONCE and return its packet count and spread."""
    packets, spreads = [], []
    for f in files:
        df = load(f)
        packets.append(len(df))
        spreads.append(spread(df))
    return packets, spreads


# ---------- 1. inspect one file per class ----------

files = {name: list_csv(folder) for name, folder in FOLDERS.items()}
first = {name: load(files[name][0]) for name in ("empty", "person")}

for name, df in first.items():
    print(f"\n=== {name}: {len(files[name])} files, first = {files[name][0]}")
    print("shape (packets, subcarriers):", df.shape)
    print("missing values:", df.isna().sum().sum())
    print(pd.DataFrame({"std": df.std(), "IQR": df.quantile(0.75) - df.quantile(0.25)}).T.round(1))


# ---------- 2. per-file statistics for every file ----------

summary = {name: summarise_folder(files[name]) for name in FOLDERS}

for name, (packets, spreads) in summary.items():
    print(f"{name:10s} files={len(spreads):3d}  median IQR min={min(spreads):6.1f}  max={max(spreads):6.1f}")
# label check: lab_static lands inside the person range, so "Static" is ambiguous -> excluded


# ---------- 3. dataset statistics table (room4 only) ----------

rows = []
for name, label in (("empty", 0), ("person", 1)):
    packets, spreads = summary[name]
    rows.append({
        "class": name,
        "label": label,
        "files": len(packets),
        "packets (total)": sum(packets),
        "duration (s, ≈)": len(packets) * FILE_S,
        "packets/s (median)": round(np.median(packets) / FILE_S),
        f"{WIN_S}-s windows (non-overlapping)": len(packets) * (FILE_S // WIN_S),
        "median IQR (min–max)": f"{min(spreads):.1f}–{max(spreads):.1f}",
    })
table = pd.DataFrame(rows)
print("\n" + table.to_string(index=False))
table.to_csv(f"{FIG_DIR}/table_dataset_stats.csv", index=False)


# ---------- 4. figures ----------

# raw subcarrier 0 over a whole file: shows the spikes in the empty room
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharey=True, constrained_layout=True)
for ax, name, title in zip(axes, ("empty", "person"),
                           ("room4, empty (Static_noperson)", "room4, 1 person moving (Dynamic_1_person)")):
    ax.plot(first[name][0])
    ax.set_title(f"{title} - subcarrier 0")
    ax.set_xlabel("packet index")
    ax.set_ylabel("amplitude")
plt.savefig(f"{FIG_DIR}/fig_raw_subcarrier0.png", dpi=150)

# (a) one 2-s window per class, each subcarrier centred on its own median
centred = {}
for name in ("empty", "person"):
    w = window(first[name], start_s=10)
    centred[name] = w - np.median(w, axis=0)     # shows VARIATION, not absolute level
lim = np.percentile(np.abs(np.vstack(list(centred.values()))), 99)   # shared colour range, ignores spikes

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
for ax, name, title in zip(axes, ("empty", "person"),
                           ("Empty room (label 0)", "One person present (label 1)")):
    c = centred[name]
    im = ax.imshow(c.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[0, WIN_S, 0, c.shape[1]])
    ax.set_title(title)
    ax.set_xlabel("time in window (s)")
axes[0].set_ylabel("subcarrier index")
fig.colorbar(im, ax=axes, label="amplitude − subcarrier median")
plt.savefig(f"{FIG_DIR}/fig_a_heatmap.png", dpi=150)

# (b) spread per recording, empty vs person
empty_spreads, person_spreads = summary["empty"][1], summary["person"][1]
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax.boxplot([empty_spreads, person_spreads])
ax.set_xticks([1, 2], [f"Empty (0)\n{len(empty_spreads)} files", f"Person (1)\n{len(person_spreads)} files"])
ax.set_ylabel("median IQR across subcarriers")
ax.set_yscale("log")          # values span ~7 to ~290 - log scale keeps both boxes readable
ax.set_title("Typical amplitude spread per recording, room4")
plt.savefig(f"{FIG_DIR}/fig_b_spread_boxplot.png", dpi=150)

plt.show()   # last line: show() blocks until the windows are closed
