import os

from helpers import list_csv, split_files

DATA = "CSV Files" if os.path.isdir("CSV Files") else "../CSV Files"
assert os.path.isdir(DATA), "CSV Files/ not found"

FOLDERS = {
    "empty": f"{DATA}/room4_iqbal/Static_noperson_channel_11_20mhz/csi_magnitude_data",
    "person": f"{DATA}/room4_iqbal/Dynamic_1_person_ch11_20mhz/csi_magnitude_data",
    "lab_static": f"{DATA}/LAB_Final Year Project/Static_Lab_ch6_20mhz/csi_magnitude_data",
}

COMPLEX_FOLDERS = {
    name: FOLDERS[name].replace("csi_magnitude_data", "csi_complex_data")
    for name in ("empty", "person")
}

files = {name: list_csv(folder) for name, folder in FOLDERS.items()}

splits = {
    "empty": split_files(files["empty"], n_train=6, n_val=2),
    "person": split_files(files["person"], n_train=51, n_val=17),
}
