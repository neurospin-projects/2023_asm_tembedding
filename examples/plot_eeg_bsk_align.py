# -*- coding: utf-8 -*-
"""
EEG markers alignement
======================

Credit: A Grigis

The goal here is to align the EEG markers with the fMRI sliding windows.
"""

import os
import glob
import mne
import numpy as np
import pandas as pd

datadir = os.getenv("TEMBEDDING_DIR")
if datadir is None:
    raise ValueError("Please specify the dataset directory in the "
                     "TEMBEDDING_DIR variable.")
win_size = 35
eegdir = os.path.join(datadir, "derivatives", "eeg_preproc_dfc")
markerdir = os.path.join(datadir, "derivatives", "eeg_markers_dfc")
for path in glob.glob(os.path.join(markerdir, "sub-*.tsv")):
    outfile = path.replace("_makers.tsv", "_synchmakers.tsv")
    if os.path.isfile(outfile):
        continue
    markers = pd.read_csv(path, sep="\t")
    basename = os.path.basename(path)
    sid, ses, cond, run = basename.split("_")[:4]
    basename = basename.replace("_cond-", "_acq-")
    basename = basename.replace("_makers.tsv", "_epo.fif")
    eegfile = os.path.join(eegdir, sid, ses, basename)
    epochs = mne.read_epochs(eegfile)
    drops = epochs.drop_log
    aligned_markers = np.empty((len(drops), markers.shape[1]))
    aligned_markers[:] = np.nan
    indices = [idx for idx in range(len(drops)) if len(drops[idx]) == 0]
    aligned_markers[indices] = markers
    aligned_markers = aligned_markers[:-(win_size - 1) // 2]
    aligned_markers = pd.DataFrame(aligned_markers, columns=markers.columns)
    aligned_markers.to_csv(outfile, sep="\t", index=False)

