# load physionet from mne
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch
from torch import nn

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset

from braindecode.models import SleepStagerChambon2018


import mne
from mne.datasets.sleep_physionet.age import fetch_data

from cmm.data import load_sleep_physionet, extract_epochs
from cmm.cmmn import CMMN

mne.set_log_level("ERROR")
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# %%


max_epochs = 100
batch_size = 128
patience = 10
sfreq = 100
lr = 0.001

results = []
for subject_target in range(len(X_all)):
    subjects = np.arange(len(X_all))
    # remove subject_target from subjects
    subjects = subjects[subjects != subject_target]
    # split data
    subjects_train, subjects_val = train_test_split(
        subjects, test_size=0.2, random_state=42
    )

    X_train = [X_all[i].astype(np.float32) for i in subjects_train]
    y_train = [y_all[i] for i in subjects_train]
    X_val = [X_all[i].astype(np.float32) for i in subjects_val]
    y_val = [y_all[i] for i in subjects_val]
    X_target = X_all[subject_target].astype(np.float32)
    y_target = y_all[subject_target]

    # train data without CMMN
    n_channels = X_train[0].shape[1]
    X_train_concat = np.concatenate(X_train, axis=0)
    y_train_concat = np.concatenate(y_train, axis=0)
    X_val_concat = np.concatenate(X_val, axis=0)
    y_val_concat = np.concatenate(y_val, axis=0)

    valid_dataset = Dataset(X_val_concat, y_val_concat)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_concat), y=y_train_concat
    )
    module = SleepStagerChambon2018(
        n_channels=n_channels,
        sfreq=sfreq,
    )
    clf = NeuralNetClassifier(
        module=module,
        max_epochs=max_epochs,
        batch_size=batch_size,
        criterion=nn.CrossEntropyLoss(
            weight=torch.Tensor(class_weights).to(device)
        ),
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        optimizer__lr=lr,
        device=device,
        train_split=predefined_split(valid_dataset),
        callbacks=[
            (
                "early_stopping",
                EarlyStopping(
                    monitor="valid_loss", patience=patience, load_best=True
                ),
            )
        ],
    )

    clf.fit(X=X_train_concat, y=y_train_concat)

    y_pred = clf.predict(X_target)
    bacc = balanced_accuracy_score(y_target, y_pred)

    results.append(
        {"subject": subject_target, "bacc": bacc, "method": "No CMMN"}
    )

    # transformed data with CMMN
    cmmn = CMMN(filter_size=128, fs=100)
    X_train_transformed = cmmn.fit_transform(X_train)
    X_val_transformed = cmmn.transform(X_val)
    X_target_transformed = cmmn.transform([X_target])[0]

    # train data with CMMN
    X_train_concat = np.concatenate(X_train_transformed, axis=0)
    X_val_concat = np.concatenate(X_val_transformed, axis=0)

    valid_dataset = Dataset(X_val_concat, y_val_concat)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_concat), y=y_train_concat
    )
    module = SleepStagerChambon2018(
        n_channels=n_channels,
        sfreq=sfreq,
    )
    clf = NeuralNetClassifier(
        module=module,
        max_epochs=max_epochs,
        batch_size=batch_size,
        criterion=nn.CrossEntropyLoss(
            weight=torch.Tensor(class_weights).to(device)
        ),
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        optimizer__lr=lr,
        device=device,
        train_split=predefined_split(valid_dataset),
        callbacks=[
            (
                "early_stopping",
                EarlyStopping(
                    monitor="valid_loss", patience=patience, load_best=True
                ),
            )
        ],
    )

    clf.fit(X=X_train_concat, y=y_train_concat)

    y_pred = clf.predict(X_target_transformed)
    bacc = balanced_accuracy_score(y_target, y_pred)

    results.append({"subject": subject_target, "bacc": bacc, "method": "CMMN"})

df = pd.DataFrame(results)
fig = plt.figure(figsize=(4, 2.5))
sns.boxplot(
    data=df,
    width=0.7,
    x="bacc",
    y="method",
    orient="h",
    palette="deep",
    showfliers=False,
)
sns.swarmplot(
    data=df,
    x="bacc",
    y="method",
    orient="h",
    color="grey",
    alpha=0.7,
    s=5,
)

plt.title("BACC for LOPO on Physionet")
plt.ylabel("")
plt.xlabel("BACC")
