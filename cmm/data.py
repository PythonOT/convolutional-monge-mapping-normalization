import os
import mne
import numpy as np


def load_sleep_physionet(
    raw_fname, annot_fname, load_eeg_only=True, crop_wake_mins=30
):
    """Load a recording from the Sleep Physionet dataset.

    Parameters
    ----------
    raw_fname : str
        Path to the .edf file containing the raw data.
    annot_fname : str
        Path to the annotation file.
    load_eeg_only : bool
        If True, only keep EEG channels and discard other modalities
        (speeds up loading).
    crop_wake_mins : float
        Number of minutes of wake events before and after sleep events.

    Returns
    -------
    mne.io.Raw :
        Raw object containing the EEG and annotations.
    """
    mapping = {
        "EOG horizontal": "eog",
        "Resp oro-nasal": "misc",
        "EMG submental": "misc",
        "Temp rectal": "misc",
        "Event marker": "misc",
    }
    exclude = mapping.keys() if load_eeg_only else ()

    raw = mne.io.read_raw_edf(raw_fname, exclude=exclude)
    annots = mne.read_annotations(annot_fname)
    raw.set_annotations(annots, emit_warning=False)
    if not load_eeg_only:
        raw.set_channel_types(mapping)

    if crop_wake_mins > 0:  # Cut start and end Wake periods
        # Find first and last sleep stages
        mask = [x[-1] in ["1", "2", "3", "4", "R"] for x in annots.description]
        sleep_event_inds = np.where(mask)[0]

        # Crop raw
        tmin = annots[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
        tmax = annots[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
        raw.crop(tmin=tmin, tmax=tmax)

    # Rename EEG channels
    ch_names = {i: i.replace("EEG ", "") for i in raw.ch_names if "EEG" in i}
    mne.rename_channels(raw.info, ch_names)

    # Save subject and recording information in raw.info
    basename = os.path.basename(raw_fname)
    subj_nb, rec_nb = int(basename[3:5]), int(basename[5])
    raw.info["subject_info"] = {"id": subj_nb, "rec_id": rec_nb}

    return raw


def extract_epochs(raw, chunk_duration=30.0):
    """Extract non-overlapping epochs from raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to be windowed.
    chunk_duration : float
        Length of a window.

    Returns
    -------
    np.ndarray
        Epoched data, of shape (n_epochs, n_channels, n_times).
    np.ndarray
        Event identifiers for each epoch, shape (n_epochs,).
    """
    annotation_desc_2_event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }

    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=chunk_duration
    )

    # create a new event_id that unifies stages 3 and 4
    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
    }

    tmax = 30.0 - 1.0 / raw.info["sfreq"]  # tmax in included
    picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False)
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        picks=picks,
        preload=True,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
    )

    return epochs.get_data(), epochs.events[:, 2] - 1
