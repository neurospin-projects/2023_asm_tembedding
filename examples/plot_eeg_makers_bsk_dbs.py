# -*- coding: utf-8 -*-
"""
EEG markers on DBS dataset
==========================

Credit: A Grigis

The goal here is to reproduce the results of the following paper using the
BSk for the downstream classification task:

Deep brain stimulation of the thalamus restores signatures of consciousness
in a nonhuman primate model, ScienceAdvances 2022.
"""

import os
import glob
import collections
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


############################################################################
# Load the dataset
# ----------------
#
# Load the DBS data.

datadir = os.getenv("TEMBEDDING_DIR")
if datadir is None:
    raise ValueError("Please specify the dataset directory in the "
                     "TEMBEDDING_DIR variable.")
rootdir = outdir = os.path.join(datadir, os.pardir, os.pardir, os.pardir)
outdir = os.path.join(rootdir, "derivatives", "eeg_markers_dfc")
assert os.path.isdir(outdir), outdir

dfc = np.load(os.path.join(datadir, "dfc.npy")).astype(np.single)
meta = pd.read_csv(os.path.join(datadir, "metadata.tsv"), sep="\t", dtype=str)
print(meta)
sequence = meta[["monkey", "session", "run", "condition"]].copy()
sequence["run"] = sequence["run"].apply(lambda x: x.replace("run", ""))
sequence["condition"].replace({"stim-on-3v": "cm3v", "stim-on-5v": "cm5v",
                               "stim-cont-on-5v": "vl5v", "stim-off": "anest",
                               "stim-cont-on-3v": "vl3v"}, inplace=True)
sequence.drop_duplicates(inplace=True)
print(sequence)
n_runs = len(sequence)
n_wins = 464
w_size = 35
dfc = dfc.reshape((-1, n_wins, 82, 82))
iu = np.triu_indices(82, k=1)
dfc = dfc[:, :, iu[0], iu[1]]
n_conns = dfc.shape[-1]
le = preprocessing.LabelEncoder()
labels = le.fit_transform(meta["matlab_kmeans_labels"].values)
subjects = meta["monkey"].values
ids = meta["unique_id"].values
conditions = le.fit_transform(meta["condition"].values)
labels = labels.reshape(-1, n_wins)
print(f"data: {dfc.shape} - labels: {labels.shape}")

mapping = pd.read_csv(os.path.join(rootdir, "derivatives", "EEG-fMRIc.tsv"),
                      sep="\t", dtype=str)
eegdir = os.path.join(rootdir, "derivatives", "eeg_markers_dfc")
data = {"monkey": [], "session": [], "run": [], "condition": [], "path": []}
for path in glob.glob(os.path.join(eegdir, "*.tsv")):
    if path.endswith("_synchmakers.tsv"):
        basename = os.path.basename(path)
        split = basename.split("_")
        data["monkey"].append(split[0].replace("sub-", ""))
        data["session"].append(split[1].replace("ses-", ""))
        data["condition"].append(split[2].replace("cond-", ""))
        data["run"].append(split[3].replace("run-", ""))
        data["path"].append(path)
data = pd.DataFrame.from_dict(data)
print(data)
print(data.groupby("monkey").describe())
print(f"conditions: {set(data.condition.values)}")

name_mapping = {
    "PowerSpectralDensity-delta": r"$\delta$",
    "PowerSpectralDensity-deltan": r"$|\delta|$",
    "PowerSpectralDensity-theta": r"$\theta$",
    "PowerSpectralDensity-thetan": r"$|\theta|$",
    "PowerSpectralDensity-alpha": r"$\alpha$",
    "PowerSpectralDensity-alphan": r"$|\alpha|$",
    "PowerSpectralDensity-beta": r"$\beta$",
    "PowerSpectralDensity-betan": r"$|\beta|$",
    "PowerSpectralDensity-gamma": r"$\gamma$",
    "PowerSpectralDensity-gamman": r"$|\gamma|$",
    "PowerSpectralDensity-summary_se": r"SE",
    "PowerSpectralDensitySummary-summary_msf": r"MSF",
    "PowerSpectralDensitySummary-summary_sef90": r"SE90",
    "PowerSpectralDensitySummary-summary_sef95": r"SE95",
    "PermutationEntropy-default": r"PE$\Theta$",
    "SymbolicMutualInformation-weighted": r"wSMI$\Theta$",
    "KolmogorovComplexity-default": r"K"
}
selected_markers = list(name_mapping.keys())
n_markers = len(selected_markers)
eeg_markers = []
for _, row in sequence.iterrows():
    match = mapping.loc[(mapping["sub"] == row.monkey) &
                        (mapping["ses EEG"] == row.session) &
                        (mapping["Run fMRI"] == row.run)]
    if len(match) == 1:
        eeg_run = match["run EEG"].item()
        match = data.loc[(data["monkey"] == row.monkey) &
                         (data["session"] == row.session) &
                         (data["condition"] == row.condition) &
                         (data["run"] == eeg_run)]
        if len(match) != 1:
            #print(row)
            #print(match)
            print("Impossible to find a match between fMRI & EEG.")
            markers = np.empty((n_wins, n_markers)) * np.nan
        else:
            markers = pd.read_csv(match.path.item(), sep="\t")
            markers = markers[selected_markers].values[(w_size) // 2 + 1: -1]
    else:
        markers = np.empty((n_wins, n_markers)) * np.nan
    eeg_markers.append(markers)
eeg_markers = np.asarray(eeg_markers)
print(f"markers: {eeg_markers.shape}")

data = {"subject": subjects, "label": labels.reshape(-1), "id": ids}
for idx, name in enumerate(name_mapping.values()):
    data[name] = eeg_markers[..., idx].reshape(-1)
df = pd.DataFrame.from_dict(data)
df.dropna(inplace=True)
print(df)
print(collections.Counter(df.label.values))


############################################################################
# Find the most important markers
# -------------------------------
#

names = df.columns.values.tolist()
groups = (df.subject + df.id).values
y = df.label.values
for key in ("subject", "label", "id"):
    names.remove(key)
X = df[names].values
print(X.shape, y.shape, groups.shape)
logo = LeavePGroupsOut(20)
train_index, test_index = next(logo.split(X, y, groups))
X_train, y_train, s_train = X[train_index], y[train_index], groups[train_index]
X_test, y_test, s_test = X[test_index], y[test_index], groups[test_index]
print(f"train: {X_train.shape}, {y_train.shape} {np.unique(y_train)} {np.unique(s_train)}")
print(f"test: {X_test.shape}, {y_test.shape} {np.unique(y_test)} {np.unique(s_test)}")
clf = ExtraTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)
print(f"train accuracy: {acc_train}")
print(f"test accuracy: {acc_test}")
with open(os.path.join(outdir, f"markers_predictions_bsk.txt"), "wt") as of:
    of.write(f"train accuracy: {acc_train}\n")
    of.write(f"test accuracy: {acc_test}")
features_importance = clf.feature_importances_
sort_indices = np.argsort(features_importance)

fig, ax = plt.subplots()
names = np.asarray(names)
y_pos = np.arange(len(names))
ax.barh(y_pos, features_importance[sort_indices], align="center")
ax.set_yticks(y_pos, labels=names[sort_indices])
ax.invert_yaxis()
ax.axvline(x=0.08, color="gray", linestyle="--")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Gini importance")
plt.savefig(os.path.join(outdir, "markers_importance_bsk.png"), dpi=150)
