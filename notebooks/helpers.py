from constants import DROP_SUBCARRIERS, FILE_S, WIN_S
import pandas as pd
import os
import glob

def list_csv(folder): #given a folder find every.csv file and return the file paths as a sorted list
    return sorted(glob.glob(folder + "/*.csv"))


def load_raw(path): #makes a dataframe out of all the files
    return pd.read_csv(path, header=None)    # no header row: the first line is data; all 49 subcarriers


def load(path): #preprocessing - drop subcarriers 0 and 48 to reduce noise
    return load_raw(path).drop(columns=DROP_SUBCARRIERS)  


def load_complex(path): 
    """Complex CSI from a csi_complex_data file; values are written like '284+410i'.
    thus it requires to convert each string to an actual python complex number, but it requires j instead of i"""
    text = pd.read_csv(path, header=None, dtype=str)
    return text.apply(lambda col: col.str.replace("i", "j").map(complex)).to_numpy()


def iqr(df): #describes the range the middle 50% of datapoints span
    return df.quantile(0.75) - df.quantile(0.25)   # one number per subcarrier


def spread(df):
    """Takes the median of the 47 IQR values - "how variable is this files signal" """
    return iqr(df).median()


def window(df, start_s, length_s=WIN_S, file_s=FILE_S):
    """Cut a length_s-second window starting at start_s, using this file's own packet rate."""
    rate = len(df) / file_s              # estimated packets per second for this file, ex: 500 packets / 70 seconds
    a = int(start_s * rate)              #row index corresponds to 10 sec mark
    b = a + int(length_s * rate)         #row index corresponds to 12 sec mark
    return df.iloc[a:b].to_numpy()           # shape: (packets, 47)


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

#split data into train/validation/test. Done by file now by row - avoids data leakage. so many files go into each 
def split_files(files, n_train, n_val):
    return {"train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:]}