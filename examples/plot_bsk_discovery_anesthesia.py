# -*- coding: utf-8 -*-
"""
BSk discovery on anesthesia dataset
===================================

Credit: A Grigis

The goal here is to run discovery-driven (only time) expermiments. The inputs
are the dyncamic functional conectivites (dFCs) computed using a sliding
windows strategy on the anesthesia dataset.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
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
# Load the anesthesia data with 7 brain states.

datadir = os.getenv("TEMBEDDING_DIR")
if datadir is None:
    raise ValueError("Please specify the dataset directory in the "
                     "TEMBEDDING_DIR variable.")
outdir = os.path.join(datadir, os.pardir, os.pardir, "cebra")
assert os.path.isdir(outdir), outdir
dataset = Dataset(
    name="anesthesia", test_size=0.1, decimate_ratio=0, random_state=42,
    datasetdir=datadir)
X_train, y_train = dataset.get_train_data()
X_test, y_test = dataset.get_test_data()
X_train = dataset.unflatten(X_train).astype(np.single)
X_test = dataset.unflatten(X_test).astype(np.single)
print(f"train: {X_train.shape}, {y_train.shape}")
print(f"test: {X_test.shape}, {y_test.shape}")


############################################################################
# Discovery-driven training
# -------------------------
#
# Use a MLP architecture and train CEBRA by using only the time.

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
                            time_delta=40)
if distance == "euclidean":
    criterion_klass = EuclideanInfoNCE
else:
    criterion_klass = CosineInfoNCE    
criterion = criterion_klass(temperature=4, beta=1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.9)
solver = SingleSessionSolver(model=model, criterion=criterion,
                             optimizer=optimizer, scheduler=scheduler)
solver.fit(loader)
device = ms_dataset.neural.device
solver.model.eval()
with torch.no_grad():
    embeddings_train = solver.model(dataset.flatten(
        torch.from_numpy(X_train).to(device))).detach().cpu().numpy()
    embeddings_test = solver.model(dataset.flatten(
        torch.from_numpy(X_test).to(device))).detach().cpu().numpy()
print(f"train embeddings: {embeddings_train.shape}")
print(f"test embeddings: {embeddings_test.shape}")

clf = LogisticRegression(max_iter=1000, random_state=42).fit(embeddings_train, y_train)
y_pred_train = clf.predict(embeddings_train)
y_pred_test = clf.predict(embeddings_test)
acc_train = accuracy_score(y_pred_train, y_train)
acc_test = accuracy_score(y_pred_test, y_test)
print(f"train accuracy: {acc_train}")
print(f"test accuracy: {acc_test}")
with open(os.path.join(outdir, f"bsk_lr-{lr}_predictions.txt"), "wt") as of:
    of.write(f"train accuracy: {acc_train}\n")
    of.write(f"test accuracy: {acc_test}")


def rotate(angle):
    ax.view_init(azim=angle)


colors = ["black", "red", "green", "blue", "purple", "yellow", "pink"]
cmap = matplotlib.colors.ListedColormap(colors)

if 1:
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

for name, embeddings, labels in (("train", embeddings_train, y_train),
                                 ("test", embeddings_test, y_test)):
    fig = plt.figure(figsize=(14, 3))
    for pos, angle in enumerate((0, 90, 180, 270)):
        ax = fig.add_subplot(1, 4, pos + 1, projection="3d")
        ax.scatter(embeddings[:, 0], embeddings[:, 1],
                   embeddings[:, 2], cmap=cmap, c=labels, s=1)
        ax.view_init(azim=angle)
        ax.set_axis_off()
    plt.savefig(os.path.join(
        outdir, f"embeddings_lr-{lr}_{name}.png"), dpi=400)

transitions_train, transitions_test = [], []
label_transitions_train, label_transitions_test = [], []
for embeddings, transitions, labels, label_transitions in (
        (embeddings_train, transitions_train, y_train, label_transitions_train),
        (embeddings_test, transitions_test, y_test, label_transitions_test)):
    z = dataset.unflatten(np.asarray(embeddings))
    y = dataset.unflatten(np.asarray(labels))
    print(z.shape, y.shape)
    choices = np.random.choice(range(len(z)), size=4, replace=False)
    print("Choices:", choices)
    for idx in choices:
        transitions.append(np.linalg.norm(z[idx, :-1] - z[idx, 1:], axis=1))
        label_transitions.append(y[idx, :-1] - y[idx, 1:])
fig, axs = plt.subplots(4, figsize=(10, 10))
colors = plt.cm.get_cmap("Set1").colors
ax = plt.gca()
transitions = np.asarray(transitions_test)
label_transitions = np.asarray(label_transitions_test)
label_transitions[label_transitions > 0] = 1
for idx, (step_dists, cond, color) in enumerate(zip(
        transitions, label_transitions, colors)):
    axs[idx].plot(step_dists, color=color, label=str(idx))
    for pos in np.argwhere(np.array(cond) != 0).squeeze():
        axs[idx].axvline(x=pos, color="gray", linestyle="--")
    axs[idx].set_ylim([0, transitions.max()])
    axs[idx].spines["right"].set_visible(False)
    axs[idx].spines["top"].set_visible(False)
fig.savefig(os.path.join(outdir, f"transitions_lr-{lr}_test.png"))
