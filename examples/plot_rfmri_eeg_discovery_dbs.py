# -*- coding: utf-8 -*-
"""
EEG markers on DBS dataset
==========================

Credit: A Grigis

The goal here is to run hypothesis-driven (time + EEK markers) expermiments.
The inputs are the dyncamic functional conectivites (dFCs) computed using a
sliding windows strategy on the DBS dataset.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from tembedding.datasets import (
    Dataset, SimpleMultiSessionDataset, MultiSessionLoader)
from tembedding.models import CEBRAModel
from tembedding.solvers import SingleSessionSolver
from tembedding.losses import EuclideanInfoNCE, CosineInfoNCE


############################################################################
# Load the dataset
# ----------------
#
# Load the DBS data.

datadir = os.getenv("TEMBEDDING_DIR")
if datadir is None:
    raise ValueError("Please specify the dataset directory in the "
                     "TEMBEDDING_DIR variable.")
rootdir = os.path.join(datadir, os.pardir, os.pardir, os.pardir)
outdir = os.path.join(rootdir, "derivatives", "cebra")
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

selected_markers = ["PowerSpectralDensitySummary-summary_sef90",
                    "PowerSpectralDensity-betan",
                    "PowerSpectralDensity-alpha"]
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

_eeg_markers = eeg_markers.reshape(-1, n_markers)
_labels = labels.reshape(-1)
X_train, y_train = dfc, _labels
X_test, y_test = dfc, _labels
print(f"train: {X_train.shape}, {y_train.shape}")
print(f"test: {X_test.shape}, {y_test.shape}")

accu = torch.from_numpy(_eeg_markers)
euclidean_dist = torch.cdist(accu, accu)
euclidean_dist = euclidean_dist.reshape((n_runs, n_wins, n_runs, n_wins))
dist_file = os.path.join(outdir, f"markers_dist.png")
if not os.path.isfile(dist_file):
    plt.figure()
    plt.imshow(euclidean_dist[50:75, :, 50:75, :].reshape(25 * n_wins, 25 * n_wins))
    plt.colorbar()
    plt.savefig(dist_file, dpi=400)
print(f"dist: {euclidean_dist.shape}- [{np.nanmin(euclidean_dist)}, "
      f"{np.nanmax(euclidean_dist)}]")

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


############################################################################
# Hypothesis-driven training
# --------------------------
#
# Use a MLP architecture and train CEBRA by using the time and EEG markers.

input_dim = X_train.shape[-1]
output_dim = 3
distance = "cosine"
lr = 3e-1
model = CEBRAModel(
    nn.Linear(input_dim, 40),
    nn.Dropout(p=0.2),
    nn.GELU(),
    nn.Linear(40, output_dim),
    num_input=input_dim,
    num_output=output_dim,
    normalize=True)
print(model)
ms_dataset = SimpleMultiSessionDataset(X_train)
n_steps = 3000
loader = MultiSessionLoader(ms_dataset, num_steps=n_steps, batch_size=1024,
                            time_delta=40, matrix_delta=0.03,
                            distance=euclidean_dist)
if distance == "euclidean":
    criterion_klass = EuclideanInfoNCE
else:
    criterion_klass = CosineInfoNCE    
criterion = criterion_klass(temperature=2, beta=1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.9)
solver = SingleSessionSolver(model=model, criterion=criterion,
                             optimizer=optimizer, scheduler=scheduler)
solver.fit(loader)
device = ms_dataset.neural.device
solver.model.eval()
with torch.no_grad():
    embeddings_train = solver.model(
        torch.from_numpy(X_train).to(device).reshape(
            -1, n_conns)).detach().cpu().numpy()
    embeddings_test = solver.model(
        torch.from_numpy(X_test).to(device).reshape(
            -1, n_conns)).detach().cpu().numpy()
print(f"train embeddings: {embeddings_train.shape}")
print(f"test embeddings: {embeddings_test.shape}")



def geog2cart(lats, lons, R=1):
    """ Converts geographic coordinates on sphere to 3D Cartesian points.

    Args:
       Latitude(s) and longitude(s) in degrees.

    Returns:
        X,Y,Z
    """
    lonr = np.deg2rad(lons)
    latr = np.deg2rad(lats)
    pts_x = R * np.cos(latr)*np.cos(lonr)
    pts_y = R * np.cos(latr)*np.sin(lonr)
    pts_z = R * np.sin(latr)
    return pts_x, pts_y, pts_z


def cart2geog(x, y, z, R=1):
    """ Converts 3D Cartesian points to geographic coordinates on sphere.

    Args:
       3D coordinates of points..

    Returns:
        Latitudes and longitudes of points in degrees.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    assert np.all(np.abs(r-R) < R/100) , "Points are not on sphere!"
    lons = np.arctan2(y, x)
    lats = np.arcsin(z/R)
    return np.rad2deg(lats), np.rad2deg(lons)


lats, lons = cart2geog(embeddings_train[:, 0], embeddings_train[:, 1],
                       embeddings_train[:, 2])
lats = lats.reshape(-1, n_wins)
lons = lons.reshape(-1, n_wins)
lats_df = pd.DataFrame(lats.T, columns=[f"{idx}" for idx in range(len(lats))])
lons_df = pd.DataFrame(lons.T, columns=[f"{idx}" for idx in range(len(lons))])
data, columns = [], []
for idx, name in enumerate(selected_markers):
    markers_df = pd.DataFrame(eeg_markers[..., idx].T,
                              columns=[f"{idx}" for idx in range(len(lats))])
    lats_corr = lats_df.corrwith(markers_df)
    lons_corr = lons_df.corrwith(markers_df)
    data.append(lats_corr.values)
    columns.append(f"latitude {name_mapping[name]}")
    data.append(lons_corr.values)
    columns.append(f"longitude {name_mapping[name]}")
df = pd.DataFrame(np.asarray(data).T, columns=columns, index=lons_corr.index)
print(df)
df = df.dropna(how="all")
print(df)
sns.heatmap(df, annot=False, cmap="summer_r")
plt.ylabel("run", fontsize=15)
plt.savefig(os.path.join(outdir, f"corr_train.png"), dpi=400)


clf = LogisticRegression(max_iter=1000, random_state=42).fit(embeddings_train, y_train)
y_pred_train = clf.predict(embeddings_train)
y_pred_test = clf.predict(embeddings_test)
acc_train = accuracy_score(y_pred_train, y_train)
acc_test = accuracy_score(y_pred_test, y_test)
print(f"train bsk accuracy: {acc_train}")
print(f"test bsk accuracy: {acc_test}")
clf = LogisticRegression(max_iter=1000, random_state=42).fit(embeddings_train, conditions)
y_pred_cond = clf.predict(embeddings_train)
acc_cond = accuracy_score(y_pred_cond, conditions)
print(f"train cond accuracy: {acc_cond}")
with open(os.path.join(outdir, f"bsk_lr-{lr}_predictions.txt"), "wt") as of:
    of.write(f"train bsk accuracy: {acc_train}\n")
    of.write(f"test bsk accuracy: {acc_test}\n")
    of.write(f"train cond accuracy: {acc_cond}")


def rotate(angle):
    ax.view_init(azim=angle)


colors = ["cyan", "red", "green", "blue", "purple", "yellow", "pink"]
cmap = matplotlib.colors.ListedColormap(colors)

if 0:
    for name, embeddings, labels in (("train", embeddings_train, y_train),
                                     ("test", embeddings_test, y_test)):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_title(f"{name} embeddings")
        ax.scatter(embeddings[:, 0], embeddings[:, 1],
                   embeddings[:, 2], cmap=cmap, c=labels, s=1)
        ax.set_axis_off()
        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 362, 2), interval=100)
        rot_animation.save(os.path.join(outdir, f"embeddings_lr-{lr}_{name}.gif"),
                           dpi=80, writer="imagemagick")

for name, embeddings, labels in (("bsk", embeddings_train, y_train),
                                 ("cond", embeddings_train, conditions)):
    fig = plt.figure(figsize=(14, 3))
    for pos, angle in enumerate((0, 90, 180, 270)):
        ax = fig.add_subplot(1, 4, pos + 1, projection="3d")
        ax.scatter(embeddings[:, 0], embeddings[:, 1],
                   embeddings[:, 2], cmap=cmap, c=labels.reshape(-1), s=1)
        ax.view_init(azim=angle)
        ax.set_axis_off()
    plt.savefig(os.path.join(
        outdir, f"embeddings_lr-{lr}_task-{name}_train.png"), dpi=400)
