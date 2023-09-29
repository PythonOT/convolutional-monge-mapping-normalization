# load physionet from mne
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal

import mne
from mne.datasets.sleep_physionet.age import fetch_data

from cmm.data import load_sleep_physionet, extract_epochs
from cmm.cmmn import CMMN
from cmm.utils import set_axe

mne.set_log_level("ERROR")

# Load data
subjects = range(5)
recordings = [1]

fnames = fetch_data(subjects=subjects, recording=recordings, on_missing="warn")
# Load recordings
raws = [load_sleep_physionet(f[0], f[1]) for f in fnames]


# Filter the data
l_freq, h_freq = None, 30

for raw in raws:
    raw.load_data().filter(l_freq, h_freq)


# Create epochs
X_all, y_all = [], []
for raw in raws:
    data, event = extract_epochs(raw)
    # Normalize the data
    data -= np.mean(data, axis=2, keepdims=True)
    data /= np.std(data, axis=2, keepdims=True)

    X_all.append(data)
    y_all.append(event)

X, y = X_all[:4], y_all[:4]
X_target, y_target = X_all[4], y_all[4]

# transformed data with CMMN

cmmn = CMMN(filter_size=128, fs=100)
X_transformed = cmmn.fit_transform(X)
X_target_transformed = cmmn.transform(X_target)

# plot transformed data

fig, axs = plt.subplots(2, 2, figsize=(7, 5))
psd_transformed = []
for i in range(4):
    X_ = np.concatenate(X_transformed[i], axis=-1)[0]
    freqs, psd_ = scipy.signal.welch(X_, nperseg=128, fs=100)
    psd_transformed.append(psd_)

psd = []
for i in range(4):
    X_ = np.concatenate(X[i], axis=-1)[0]
    freqs, psd_ = scipy.signal.welch(X_, nperseg=128, fs=100)
    psd.append(psd_)

for i in range(4):
    axs[i // 2, i % 2].plot(freqs, psd[i], label="Source")
    axs[i // 2, i % 2].plot(freqs, psd_transformed[i], label="Transformed")
    axs[i // 2, i % 2].plot(
        freqs, cmmn.barycenter[0], color="black",
        linestyle="--", label="Barycenter"
    )
    axs[i // 2, i % 2].legend()
    axs[i // 2, i % 2].set_title(f"Subject {i}")
    set_axe(axs[i // 2, i % 2])

plt.plot()
