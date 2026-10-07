"""Stage 2: logistic regression for Wi-Fi CSI presence detection (room4, empty vs one person).

Each data point is a 2-second window (about 154 packets x 47 subcarriers) summarised into a
feature vector of per-subcarrier statistics; the label is 0 = empty room, 1 = person present.
Two feature sets are compared on the validation set:
  "median": IQR + median per subcarrier                       -> 94 features
  "change": IQR + median absolute change between packets         -> 94 features
Run from the repo root; the data is expected in "CSV Files/" (same as main.py).
"""
import glob
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

# numpy 2.0 + macOS Accelerate raises spurious "encountered in matmul" warnings (results are unaffected)
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# ---------- Setup ----------

DATA = "CSV Files" if os.path.isdir("CSV Files") else "../CSV Files"
assert os.path.isdir(DATA), "CSV Files/ not found"

FOLDERS = {
    "empty":  f"{DATA}/room4_iqbal/Static_noperson_channel_11_20mhz/csi_magnitude_data",
    "person": f"{DATA}/room4_iqbal/Dynamic_1_person_ch11_20mhz/csi_magnitude_data",
}
LABELS = {"empty": 0, "person": 1}
DROP_SUBCARRIERS = [0, 48]   # band edges, noisier in every empty recording (Stage 1, appendix 1.5)
FILE_S = 70    # seconds between consecutive file start times (from the filenames)
WIN_S = 2      # window length in seconds


# ---------- Helpers (same as main.py) ----------

def list_csv(folder):
    return sorted(glob.glob(folder + "/*.csv"))


def load(path):
    df = pd.read_csv(path, header=None)       # no header row: the first line is data
    return df.drop(columns=DROP_SUBCARRIERS)  # 49 -> 47 subcarriers (original indices kept)


def window(df, start_s, length_s=WIN_S, file_s=FILE_S):
    """Cut a length_s-second window starting at start_s, using this file's own packet rate."""
    rate = len(df) / file_s                      # estimated packets per second for this file
    a = int(start_s * rate)
    b = a + int(length_s * rate)
    return df.iloc[a:b].to_numpy()               # shape: (packets, 47)


def split_files(files, n_train, n_val):
    return {"train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:]}

# ---------- Step 3: features for one window ----------

def window_features(w, kind):
    """Feature vector for one window.

    w:    numpy array of shape (packets, 47), one 2-second window
    kind: "median" -> IQR + median per subcarrier
          "change" -> IQR + median absolute change between consecutive packets
    returns a 1-D array of length 94
    """
    IQR = np.percentile(w, 75, axis=0) - np.percentile(w, 25, axis=0)
    if kind == "median":
       second_array = np.median(w, axis=0)
    elif kind == "change":
       second_array = np.median(np.abs(np.diff(w, axis=0)), axis=0)
    else:
        raise ValueError("Unknown kind")
    return np.concatenate([IQR, second_array], axis=0)


# ---------- one window per class ----------

files = {name: list_csv(folder) for name, folder in FOLDERS.items()}
assert len(files["empty"]) == 10 and len(files["person"]) == 85, {n: len(f) for n, f in files.items()}

for name in ("empty", "person"):
    w = window(load(files[name][0]), start_s=10)
    for kind in ("median", "change"):
        f = window_features(w, kind)
        print(f"{name:6s} {kind:6s} shape={f.shape}  IQR[:5]={np.round(f[:5], 1)}  {kind}[:5]={np.round(f[47:52], 1)}")


# ----------  X, y for train / val / test ----------

def build_set(part, kind):
    splits = {"empty": split_files(files["empty"], 6, 2),
              "person": split_files(files["person"], 51, 17)}

    X, y = [], []
    for label in ("empty", "person"):
        for path in splits[label][part]:  # each part recording of this class
            df = load(path)
            for start_s in range(0, FILE_S, WIN_S):  # 0, 2, ..., 68 -> 35 windows
                X.append(window_features(window(df, start_s), kind))
                y.append(LABELS[label])
    return np.array(X), np.array(y)

X_train_median, y_train_median = build_set("train", "median")
X_val_median, y_val_median = build_set("val", "median")
X_test_median, y_test_median = build_set("test", "median")

X_train_change, y_train_change = build_set("train", "change")
X_val_change, y_val_change = build_set("val", "change")
X_test_change, y_test_change = build_set("test", "change")
# ---------- Step 5: standardise with training statistics only ----------
mean_1 = X_train_median.mean(axis=0)
std_1 = X_train_median.std(axis=0)
mean_2 = X_train_change.mean(axis=0)
std_2 = X_train_change.std(axis=0)

X_train_median_standardised = (X_train_median - mean_1) / std_1
X_val_median_standardised = (X_val_median - mean_1) / std_1
X_test_median_standardised = (X_test_median - mean_1) / std_1

X_train_change_standardised = (X_train_change - mean_2) / std_2
X_val_change_standardised = (X_val_change - mean_2) / std_2
X_test_change_standardised = (X_test_change - mean_2) / std_2

# ----------  train logistic regression (class weights) and tune lambda on validation ----------
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]   # log-spaced candidates


def tune_C(X_train, y_train, X_val, y_val, name):
    """Fit one model per C on the training set, score it on the validation set, return the best C."""
    best_C, best_score = None, -1.0
    for C in C_VALUES:
        logistic_regression = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
        logistic_regression.fit(X_train, y_train)
        y_pred = logistic_regression.predict(X_val)
        score = balanced_accuracy_score(y_val, y_pred)
        if score > best_score:
            best_C = C
            best_score = score
        print(f"{name:6s}  C={C:<7}  val balanced accuracy={score}")
    print(f"{name:6s}  best C = {best_C}  (val balanced accuracy {best_score})\n")
    return best_C


best_C_median = tune_C(X_train_median_standardised, y_train_median,
                       X_val_median_standardised, y_val_median, "median")
best_C_change = tune_C(X_train_change_standardised, y_train_change,
                       X_val_change_standardised, y_val_change, "change")

# ---------- Sanity check: shuffled training labels ----------
# If the pipeline is correct, a model trained on randomly shuffled labels can learn nothing real,
# so its validation balanced accuracy should drop to about 0.5 (chance). A score near 1.0 here
# would mean information about the validation labels leaks into training somewhere.

def shuffled_label_check(X_train, y_train, X_val, y_val, C, name, n_runs=5):
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(n_runs):
        y_shuffled = rng.permutation(y_train)            # same class counts, labels no longer match the windows
        logistic_regression = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
        logistic_regression.fit(X_train, y_shuffled)
        scores.append(balanced_accuracy_score(y_val, logistic_regression.predict(X_val)))
    print(f"{name:6s}  shuffled labels, C={C}: val balanced accuracy "
          f"mean {np.mean(scores):.3f} (runs: {', '.join(f'{s:.3f}' for s in scores)})")


shuffled_label_check(X_train_median_standardised, y_train_median,
                     X_val_median_standardised, y_val_median, best_C_median, "median")
shuffled_label_check(X_train_change_standardised, y_train_change,
                     X_val_change_standardised, y_val_change, best_C_change, "change")


# ---------- Weight check: what does each model rely on? ----------
# coef_[0] holds the 94 weights in feature order: [0:47] = IQR, [47:94] = second statistic.
# The features are standardised, so a larger |weight| means the feature matters more.

def final_model(X_train, y_train, C):
    """Train the model that will be evaluated in the last step (best C, training set only)."""
    logistic_regression = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    return logistic_regression.fit(X_train, y_train)


def weight_summary(model, name):
    w = model.coef_[0]
    iqr_importance = np.abs(w[:47]).mean()
    second_importance = np.abs(w[47:]).mean()
    second_sign = np.mean(w[47:])
    print(f"{name:6s}  mean |w| IQR = {iqr_importance}   mean |w| {name} = {second_importance}   "
          f"mean w {name} = {second_sign}")


final_median = final_model(X_train_median_standardised, y_train_median, best_C_median)
final_change = final_model(X_train_change_standardised, y_train_change, best_C_change)
weight_summary(final_median, "median")
weight_summary(final_change, "change")


# ---------- IQR-only model (ablation: does the second statistic add anything?) ----------
# Both feature sets start with the same 47 IQR columns, so the IQR-only features are just the first
# 47 columns; slicing the already-standardised arrays gives the same result as standardising them alone.
X_train_iqr_standardised = X_train_change_standardised[:, :47]
X_val_iqr_standardised   = X_val_change_standardised[:, :47]
X_test_iqr_standardised  = X_test_change_standardised[:, :47]

best_C_iqr = tune_C(X_train_iqr_standardised, y_train_change, X_val_iqr_standardised, y_val_change, "iqr")
final_iqr = final_model(X_train_iqr_standardised, y_train_change, best_C_iqr)

for part, X, y in (("val", X_val_iqr_standardised, y_val_change), ("test", X_test_iqr_standardised, y_test_change)):
    pred = final_iqr.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"iqr     {part:4s}  accuracy = {np.mean(pred == y):.4f}   balanced accuracy = {balanced_accuracy_score(y, pred):.4f}"
          f"   false alarms = {fp}/{tn + fp}   missed = {fn}/{fn + tp}")
print()


# ---------- Training / validation / test errors (for the Results section) ----------
# Two error measures, the same for every set:
#   log loss       = class-weighted logistic loss, the loss the model is trained on (data term only, no regularisation)
#   balanced error = 1 - balanced accuracy, the 0/1 error averaged per class (fair under the 8.5 : 1 imbalance)
# The final method is chosen from the train/val rows; only the chosen model's test row goes into the report as its test error.

def report_errors(model, sets, name):
    """sets: dict like {"train": (X, y), "val": (X, y), "test": (X, y)}, X already standardised."""
    for part, (X, y) in sets.items():
        weights = compute_sample_weight("balanced", y)   # one weight per window: empty ~4.75, person ~0.56
        p = model.predict_proba(X)[:, 1]                 # predicted probability of "person"
        loss = log_loss(y, p, sample_weight=weights)
        b_err = 1 - balanced_accuracy_score(y, model.predict(X))
        print(f"{name:6s} {part:5s}  log loss = {loss:.4f}   balanced error = {b_err:.4f}")
    print()


report_errors(final_median, {"train": (X_train_median_standardised, y_train_median),
                             "val":   (X_val_median_standardised,   y_val_median),
                             "test":  (X_test_median_standardised,  y_test_median)}, "median")
report_errors(final_change, {"train": (X_train_change_standardised, y_train_change),
                             "val":   (X_val_change_standardised,   y_val_change),
                             "test":  (X_test_change_standardised,  y_test_change)}, "change")
report_errors(final_iqr,    {"train": (X_train_iqr_standardised,    y_train_change),
                             "val":   (X_val_iqr_standardised,      y_val_change),
                             "test":  (X_test_iqr_standardised,     y_test_change)}, "iqr")


# ---------- evaluate the FINAL model on the test set (confusion matrix, TPR, TNR, precision, F1, ROC/AUC) ----------
# Final choice, made on validation only: all three LR models tie at 0 validation balanced error, the CNN has 0.0008,
# and among the tied models the simplest one is chosen -> LR_IQR (47 IQR features, 48 parameters).
# The windows and labels are the same for every feature set, so y_test_change is the test label array.
final_model_chosen, X_test_chosen, y_test = final_iqr, X_test_iqr_standardised, y_test_change

y_pred = final_model_chosen.predict(X_test_chosen)
y_prob = final_model_chosen.predict_proba(X_test_chosen)[:, 1]          # P(person present), for ROC-AUC
score = balanced_accuracy_score(y_test, y_pred)

# confusion_matrix rows = true class, columns = predicted class, in label order [0 = empty, 1 = person]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

print("\n=== Test set, final model: LR_IQR (IQR features only), C =", best_C_iqr, "===")
print(f"windows: {len(y_test)}  (empty {np.sum(y_test == 0)}, person {np.sum(y_test == 1)})")
print("confusion matrix (rows = true, columns = predicted):")
print(f"                 pred empty  pred person")
print(f"  true empty     {tn:10d}  {fp:11d}   <- FP = false alarms")
print(f"  true person    {fn:10d}  {tp:11d}   <- FN = missed intruders")
print(f"TPR (recall, person detected)       = {recall_score(y_test, y_pred, pos_label=1):.3f}")
print(f"TNR (empty correctly left alone)     = {recall_score(y_test, y_pred, pos_label=0):.3f}")
print(f"precision (alarms that were real)    = {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.3f}")
print(f"F1 (person class)                    = {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.3f}")
print(f"balanced accuracy = (TPR + TNR) / 2  = {score:.3f}")
print(f"balanced error = 1 - balanced acc.   = {1 - score:.4f}")
print(f"test log loss (class-weighted)       = {log_loss(y_test, y_prob, sample_weight=compute_sample_weight('balanced', y_test)):.4f}")
print(f"ROC-AUC                              = {roc_auc_score(y_test, y_prob):.3f}")
