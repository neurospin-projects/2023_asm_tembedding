# -*- coding: utf-8 -*-
"""
EEG markers on DBS dataset
==========================

Credit: A Grigis

The goal here is to reproduce the results of the following paper:

Deep brain stimulation of the thalamus restores signatures of consciousness
in a nonhuman primate model, ScienceAdvances 2022.
"""

import os
import glob
import mne
import numpy as np
import pandas as pd
from mulm.residualizer import Residualizer
from sklearn import preprocessing
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from tembedding.eeg import eeg_markers


############################################################################
# Fetch the data
# --------------
#
# The data were preprocessed as follows:
# Following the MR Ballisto-CardioGram and motion artifacts cleaning, the
# signal was filtered between 1 Hz (IIR Butterworth high-pass zero-phase
# two-pass forward and reverse noncausal filter, order 12—effective, after
# forward-backward) and 25 Hz (IIR  Butterworth  lowpass  zero-phase
# noncausal filter, order 16—effective, after forward-backward) and
# downsampled to 250-Hz sampling rate. The data were cut 15 s after the
# start of the stimulation until 15 s before the end of the scanning. The
# remaining time series was cut into epochs of 0.8 s, with a random jitter
# ranging from  0.55  to  0.85 s. For the cleaning of artifacted epochs or
# channels, the Python package Autoreject was used with the number of
# channel interpolations equal to 1, 2, 4, or 8. Average EEG reference
# projection was applied.


datadir = os.getenv("TEMBEDDING_DIR")
if datadir is None:
    raise ValueError("Please specify the dataset directory in the "
                     "TEMBEDDING_DIR variable.")
dataset = "dfc"
data = {"subject": [], "session": [], "run": [], "condition": [], "path": []}
if dataset == "paper":
    eegdir = os.path.join(datadir, "derivatives", "eeg_preproc")
    for path in glob.glob(os.path.join(eegdir, "*", "*", "*.fif")):
        split = path.split(os.sep)
        data["condition"].append(split[-3].replace(
            "_results", "").replace("_", ""))
        sid, ses = split[-2].split("_")[:2]
        data["subject"].append(sid)
        data["session"].append(ses)
        data["run"].append(split[-2].split("_")[-1])
        data["path"].append(path)
else:
    eegdir = os.path.join(datadir, "derivatives", "eeg_preproc_dfc")
    for path in glob.glob(os.path.join(eegdir, "*", "*", "*.fif")):
        basename = os.path.basename(path)
        sid, ses, cond, run, _ = basename.split("_")
        data["subject"].append(sid.replace("sub-", ""))
        data["session"].append(ses.replace("ses-", ""))
        data["condition"].append(cond.replace("acq-", ""))
        data["run"].append(run.replace("run-", ""))
        data["path"].append(path)
data = pd.DataFrame.from_dict(data)
print(data)
print(data.groupby("subject").describe())
print(f"conditions: {set(data.condition.values)}")


############################################################################
# Compute the dynamic EEG markers
# -------------------------------
#

if eegdir.endswith("_dfc"):
    outdir = os.path.join(datadir, "derivatives", "eeg_markers_dfc")
else:
    outdir = os.path.join(datadir, "derivatives", "eeg_markers")
if not os.path.isdir(outdir):
    os.mkdir(outdir)
compute_markers = False
if compute_markers:
    Parallel(n_jobs=-2)(delayed(eeg_markers)(
            epochs_file=row.path, outdir=outdir, njobs=1,
            basename=(f"sub-{row.subject}_ses-{row.session}_cond-{row.condition}_"
                      f"run-{row.run}"))
        for _, row in data.iterrows())
data["marker_path"] = [os.path.join(
    outdir, (f"sub-{row.subject}_ses-{row.session}_cond-{row.condition}_"
             f"run-{row.run}_makers.tsv"))
    for _, row in data.iterrows()]
print(data)


############################################################################
# Plot the EEG markers
# --------------------
#

markers = {"condition": [], "subject": [], "session": []}
full_markers = {"condition": [], "subject": [], "session": []}
for idx, row in data.iterrows():
    _data = pd.read_csv(row.marker_path, sep="\t")
    for _key in _data.columns:
        _mean = np.mean(_data[_key].values)
        markers.setdefault(_key, []).append(_mean)
        full_markers.setdefault(_key, []).extend(_data[_key].values.tolist())
    markers["condition"].append(row.condition)
    markers["subject"].append(row.subject + row.session + row.run)
    markers["session"].append(idx)
    full_markers["condition"].extend([row.condition] * len(_data))
    full_markers["subject"].extend([row.subject + row.session + row.run] * len(_data))
    full_markers["session"].extend([idx] * len(_data))
display_markers = False
if display_markers:
    sns.set_style("white")
    palette = "Set2"
    marker_names = list(markers.keys())
    marker_names.remove("condition")
    for marker_name in marker_names:
        plt.figure()
        ax = sns.violinplot(x="condition", y=marker_name, data=markers,
                            hue="condition", dodge=False, palette=palette,
                            scale="width", inner=None)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for violin in ax.collections:
            bbox = violin.get_paths()[0].get_extents()
            x0, y0, width, height = bbox.bounds
            violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height,
                                 transform=ax.transData))
        sns.boxplot(x="condition", y=marker_name, data=markers, saturation=1,
                    showfliers=False, width=0.3,
                    boxprops={"zorder": 3, "facecolor": "none"}, ax=ax)
        old_len_collections = len(ax.collections)
        sns.stripplot(x="condition", y=marker_name, data=markers,
                      hue="condition", palette=palette, dodge=False, ax=ax)
        for dots in ax.collections[old_len_collections:]:
            dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend_.remove()
        ax.spines[["right", "left", "top"]].set_visible(False)
        ax.grid(axis="y")
        plt.savefig(os.path.join(outdir, f"{marker_name}.png"), dpi=150)
print(markers.keys())


############################################################################
# Find the most important markers
# -------------------------------
#

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

markers = (full_markers if True else markers)
groups = np.asarray(markers["subject"])
sessions = np.asarray(markers["session"])
le = preprocessing.LabelEncoder()
conditions = np.asarray(markers["condition"])
keep_indices = np.argwhere(np.isin(conditions, ("cm3v", "cm5v", "anest")))
conditions = np.asarray([
    name if name not in ("vl3v", "vl5v") else "anest" for name in conditions])
conditions = conditions[keep_indices].squeeze()
groups = groups[keep_indices].squeeze()
sessions = sessions[keep_indices].squeeze()
print(f"conditions: {np.unique(conditions)}")
y = le.fit_transform(conditions)
names = list(markers.keys())
for key in ("condition", "subject", "session"):
    names.remove(key)
X = np.array([markers[name] for name in names]).T
X = X[keep_indices].squeeze()
if False:
    data = pd.DataFrame.from_dict({"group": sessions})
    res = Residualizer(data=data, formula_res="group")
    design = res.get_design_mat(data)
    X = res.fit_transform(X, design)
names = np.asarray([name_mapping[name] for name in names])
logo = LeavePGroupsOut(20)
print(X.shape, y.shape, groups.shape)
train_index, test_index = next(logo.split(X, y, groups))
X_train, y_train, s_train = X[train_index], y[train_index], groups[train_index]
X_test, y_test, s_test = X[test_index], y[test_index], groups[test_index]
print(f"train: {X_train.shape}, {y_train.shape} {np.unique(y_train)} {np.unique(s_train)}")
print(f"test: {X_test.shape}, {y_test.shape} {np.unique(y_test)} {np.unique(s_test)}")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
for idx, feature_name in enumerate(names):
    print("-" * 10)
    print(f"feature: {feature_name}")
    clf = ExtraTreeClassifier(random_state=42)
    clf.fit(X_train[:, [idx]], y_train)
    acc_train = clf.score(X_train[:, [idx]], y_train)
    acc_test = clf.score(X_test[:, [idx]], y_test)
    print(f"train accuracy: {acc_train}")
    print(f"test accuracy: {acc_test}")
print("-" * 10)
print("feature: multi")
clf = ExtraTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)
print(f"train accuracy: {acc_train}")
print(f"test accuracy: {acc_test}")
with open(os.path.join(outdir, f"markers_predictions.txt"), "wt") as of:
    of.write(f"train accuracy: {acc_train}\n")
    of.write(f"test accuracy: {acc_test}")
features_importance = clf.feature_importances_
sort_indices = np.argsort(features_importance)

fig, ax = plt.subplots()
y_pos = np.arange(len(names))
ax.barh(y_pos, features_importance[sort_indices], align="center")
ax.set_yticks(y_pos, labels=names[sort_indices])
ax.invert_yaxis()
ax.axvline(x=0.08, color="gray", linestyle="--")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Gini importance")
plt.savefig(os.path.join(outdir, "markers_importance.png"), dpi=150)
