"""Stage 1 exploratory analysis: Wi-Fi CSI presence detection (room4, empty vs one person).

Script version of notebooks/stage1_eda.ipynb (same code, same outputs); run from the repo root.
Data: IEEE DataPort "CSI Dataset for WiFi Based Human Detection" (DOI 10.21227/smn7-jb67),
unzipped into the project as "CSV Files/". Each magnitude CSV = one ~70 s recording,
rows = packets (time), 49 columns = subcarriers (47 used), no header, no timestamps.
Figures and tables are written to figures/.
"""
# ---------- Setup and data location ----------


import glob
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# find the data: unzip any data_*.zip next to the notebook (JupyterHub), else use ../CSV Files (local repo)
for zip_path in sorted(glob.glob("data_*.zip")):
    with zipfile.ZipFile(zip_path) as z:
        if not os.path.exists(z.namelist()[-1]):     # not extracted yet
            z.extractall(".")
DATA = "CSV Files" if os.path.isdir("CSV Files") else "../CSV Files"
assert os.path.isdir(DATA), "CSV Files/ not found - upload data_room4.zip next to this notebook"

# magnitude files only: the csi_complex_data/ folders (amplitude + phase) are not used,
# so CSI phase is never loaded
FOLDERS = {
    "empty":      f"{DATA}/room4_iqbal/Static_noperson_channel_11_20mhz/csi_magnitude_data",
    "person":     f"{DATA}/room4_iqbal/Dynamic_1_person_ch11_20mhz/csi_magnitude_data",
    # not used for training: only checked to decide what "Static" means
    "lab_static": f"{DATA}/LAB_Final Year Project/Static_Lab_ch6_20mhz/csi_magnitude_data",
}
# complex (amplitude + phase) files: only read in the phase check below, never used as features
COMPLEX_FOLDERS = {name: FOLDERS[name].replace("csi_magnitude_data", "csi_complex_data")
                   for name in ("empty", "person")}
N_PHASE_FILES = 10   # first 10 recordings per class are enough to show the pattern
EXPECTED_FILES = {"empty": 10, "person": 85, "lab_static": 91}
DROP_SUBCARRIERS = [0, 48]   # band edges, see "Subcarrier selection" below
FILE_S = 70    # seconds between consecutive file start times (from the filenames)
WIN_S = 2      # window length in seconds
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ---------- Helpers ----------


def list_csv(folder):
    return sorted(glob.glob(folder + "/*.csv"))


def load_raw(path):
    return pd.read_csv(path, header=None)    # no header row: the first line is data; all 49 subcarriers


def load(path):
    return load_raw(path).drop(columns=DROP_SUBCARRIERS)   # 49 -> 47 subcarriers (original indices kept)


def load_complex(path):
    """Complex CSI from a csi_complex_data file; values are written like '284+410i'."""
    text = pd.read_csv(path, header=None, dtype=str)
    return text.apply(lambda col: col.str.replace("i", "j").map(complex)).to_numpy()


def iqr(df):
    return df.quantile(0.75) - df.quantile(0.25)   # one number per subcarrier


def spread(df):
    """Median over subcarriers of the per-subcarrier IQR."""
    return iqr(df).median()


def window(df, start_s, length_s=WIN_S, file_s=FILE_S):
    """Cut a length_s-second window starting at start_s, using this file's own packet rate."""
    rate = len(df) / file_s                      # estimated packets per second for this file
    a = int(start_s * rate)
    b = a + int(length_s * rate)
    return df.iloc[a:b].to_numpy()               # shape: (packets, 47)


def summarise_folder(files):
    """Load each file ONCE and return its packet count and spread."""
    packets, spreads = [], []
    for f in files:
        df = load(f)
        packets.append(len(df))
        spreads.append(spread(df))
    return packets, spreads


def start_time(path):
    """Recording start time from a filename like csi_20241204_020632_magnitude.csv."""
    stamp = os.path.basename(path).split("_")[1:3]            # ['20241204', '020632']
    return pd.to_datetime("_".join(stamp), format="%Y%m%d_%H%M%S")


def split_files(files, n_train, n_val):
    return {"train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:]}


# ---------- Inspect one file per class ----------


files = {name: list_csv(folder) for name, folder in FOLDERS.items()}
print({name: len(f) for name, f in files.items()})
# a half-finished unzip leaves too few files; if this fails, delete "CSV Files/" and re-run from the top
for name, n in EXPECTED_FILES.items():
    assert len(files[name]) in (n, 0 if name == "lab_static" else n), (name, len(files[name]))

first = {name: load(files[name][0]) for name in ("empty", "person")}

for name, df in first.items():
    print(f"\n=== {name}: {len(files[name])} files, first = {os.path.basename(files[name][0])}")
    print("shape (packets, subcarriers):", df.shape)
    print("missing values:", df.isna().sum().sum())
    print(pd.DataFrame({"std": df.std(), "IQR": iqr(df)}).T.round(1))


# ---------- Raw signal of one subcarrier over a whole recording ----------


fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharey=True, constrained_layout=True)
for ax, name, title in zip(axes, ("empty", "person"),
                           ("room4, empty (Static_noperson)", "room4, 1 person moving (Dynamic_1_person)")):
    ax.plot(first[name][1])
    ax.set_title(f"{title} - subcarrier 1")
    ax.set_xlabel("packet index")
    ax.set_ylabel("amplitude")
plt.savefig(f"{FIG_DIR}/fig_raw_subcarrier1.png", dpi=150)


# ---------- Subcarrier selection: band edges ----------


ratio_rows = []
for name in ("empty", "person"):
    ratios = []
    for f in files[name]:
        r = iqr(load_raw(f))
        inner = r.drop(index=DROP_SUBCARRIERS).median()
        ratios.append((r[0] / inner, r[48] / inner))
    ratios = np.array(ratios)
    for i, sc in enumerate(DROP_SUBCARRIERS):
        ratio_rows.append({"class": name, "subcarrier": sc, "files": len(ratios),
                           "files with ratio > 1": int((ratios[:, i] > 1).sum()),
                           "median ratio": np.median(ratios[:, i]),
                           "min": ratios[:, i].min(), "max": ratios[:, i].max()})
edge_table = pd.DataFrame(ratio_rows).round(2)
edge_table.to_csv(f"{FIG_DIR}/table_edge_subcarriers.csv", index=False)
print("\n" + edge_table.to_string(index=False))


# ---------- Feature selection: why phase is not used ----------


phase_rows = []
for name, folder in COMPLEX_FOLDERS.items():
    complex_files = list_csv(folder)[:N_PHASE_FILES]
    if not complex_files:
        print(f"{name}: no complex files found - upload data_room4_phase.zip to run this check")
        continue
    circ_var, jump, mag_rel = [], [], []
    for f in complex_files:
        c = load_complex(f)[:, 1:48]                    # inner subcarriers, same as the features
        phase, mag = np.angle(c), np.abs(c)
        circ_var.append(np.median(1 - np.abs(np.mean(np.exp(1j * phase), axis=0))))
        jump.append(np.median(np.abs(np.angle(np.exp(1j * np.diff(phase, axis=0))))))
        q75, q25 = np.percentile(mag, [75, 25], axis=0)
        mag_rel.append(np.median((q75 - q25) / np.median(mag, axis=0)))
    phase_rows.append({"class": name, "files": len(complex_files),
                       "phase circular variance": np.median(circ_var),
                       "median phase jump (rad)": np.median(jump),
                       "magnitude IQR / median": np.median(mag_rel)})
phase_table = pd.DataFrame(phase_rows).round(3)
phase_table.to_csv(f"{FIG_DIR}/table_phase_check.csv", index=False)
print("\n" + phase_table.to_string(index=False))


# ---------- Spread per recording, every file ----------


summary = {name: summarise_folder(files[name]) for name in FOLDERS if files[name]}

for name, (packets, spreads) in summary.items():
    print(f"{name:10s} files={len(spreads):3d}  median IQR min={min(spreads):6.1f}  max={max(spreads):6.1f}")

# label check: if the Lab "Static" folder was loaded, compare its spread with room4
if "lab_static" in summary:
    lab = summary["lab_static"][1]
    print(f"\nLab 'Static' range {min(lab):.1f}-{max(lab):.1f} vs room4 empty max {max(summary['empty'][1]):.1f}, "
          f"person min {min(summary['person'][1]):.1f} -> inside the person range, so 'Static' is ambiguous "
          f"(empty in a noisier room, or a still person) and the Lab folders are excluded")


# ---------- Dataset statistics for room 4 ----------


rows = []
for name, label in (("empty", 0), ("person", 1)):
    packets, spreads = summary[name]
    rows.append({
        "class": name,
        "label": label,
        "files": len(packets),
        "packets (total)": sum(packets),
        "duration (s, approx.)": len(packets) * FILE_S,
        "packets/s (median)": round(np.median(packets) / FILE_S),
        f"{WIN_S}-s windows (non-overlapping)": len(packets) * (FILE_S // WIN_S),
        "median IQR (min-max)": f"{min(spreads):.1f}-{max(spreads):.1f}",
    })
table = pd.DataFrame(rows)
table.to_csv(f"{FIG_DIR}/table_dataset_stats.csv", index=False)
print("\n" + table.to_string(index=False))


# ---------- Train / validation / test split ----------


splits = {
    "empty":  split_files(files["empty"],  n_train=6,  n_val=2),    # 6 / 2 / 2
    "person": split_files(files["person"], n_train=51, n_val=17),   # 51 / 17 / 17
}

split_rows = []
for name, parts in splits.items():
    for part, part_files in parts.items():
        split_rows.append({
            "class": name,
            "set": part,
            "files": len(part_files),
            f"{WIN_S}-s windows": len(part_files) * (FILE_S // WIN_S),
            "first start": start_time(part_files[0]),
            "last start": start_time(part_files[-1]),
        })
split_table = pd.DataFrame(split_rows)
split_table.to_csv(f"{FIG_DIR}/table_split.csv", index=False)

# no file may appear in two sets
all_split = [f for parts in splits.values() for part_files in parts.values() for f in part_files]
assert len(all_split) == len(set(all_split)) == len(files["empty"]) + len(files["person"])
print("\n" + split_table.to_string(index=False))


# ---------- Figure (a): one 2-s window per class ----------


centred = {}
for name in ("empty", "person"):
    w = window(first[name], start_s=10)
    centred[name] = w - np.median(w, axis=0)     # shows variation
lim = np.percentile(np.abs(np.vstack(list(centred.values()))), 99)   # shared colour range, ignores spikes

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
for ax, name, title in zip(axes, ("empty", "person"),
                           ("Empty room (label 0)", "One person present (label 1)")):
    c = centred[name]
    im = ax.imshow(c.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[0, WIN_S, 1, 48])
    ax.set_title(title)
    ax.set_xlabel("time in window (s)")
axes[0].set_ylabel("subcarrier index (0 and 48 dropped)")
fig.colorbar(im, ax=axes, label="amplitude - subcarrier median")
plt.savefig(f"{FIG_DIR}/fig_a_heatmap.png", dpi=150)


# ---------- Figure (b): spread per recording, empty vs person ----------


empty_spreads, person_spreads = summary["empty"][1], summary["person"][1]
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax.boxplot([empty_spreads, person_spreads])
ax.set_xticks([1, 2], [f"Empty (0)\n{len(empty_spreads)} files", f"Person (1)\n{len(person_spreads)} files"])
ax.set_ylabel("median IQR across subcarriers")
ax.set_yscale("log")          # values span about 7 to 300
ax.set_title("Typical amplitude spread per recording, room4")
plt.savefig(f"{FIG_DIR}/fig_b_spread_boxplot.png", dpi=150)

plt.show()   # last line: show() blocks until the windows are closed
