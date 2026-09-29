N_PHASE_FILES = 10   # first 10 recordings per class are enough to show the pattern
EXPECTED_FILES = {"empty": 10, "person": 85, "lab_static": 91} #how many files should be in each folder
DROP_SUBCARRIERS = [0, 48]   # band edges (noisy, so drop), see "Subcarrier selection" below
FILE_S = 70    # seconds between consecutive file start times (from the filenames)
WIN_S = 2      # window length in seconds
FIG_DIR = "figures"