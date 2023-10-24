# Convolutional Monge Mapping Normalization for learning on Sleep data

This repository is the official implementation of Convolutional Monge Mapping Normalization for learning on Sleep data. This repository proposes an example of the method applied to Physionet data available on [MNE-python](https://mne.tools/stable/index.html). 

(The other datasets used in the paper are available on request. That the reason
why we propose an example on Physionet data only.)

The reference paper is available on [arXiv](https://arxiv.org/pdf/2305.18831.pdf).
Please cite the paper if you use this code in your research.

```
T. Gnassounou, R. Flamary, A. Gramfort, Convolutional Monge Mapping Normalization for learning on biosignals, Neural Information Processing Systems (NeurIPS), 2023.
```

Bibtex entry:

```
@inproceedings{gnassounou2023convolutional,
author = {Gnassounou, Théo and Flamary, Rémi and Gramfort, Alexandre},
title = {Convolutional Monge Mapping Normalization for learning on biosignals},
booktitle = {Neural Information Processing Systems (NeurIPS)},
year = {2023}
}
```


### Install pyTorch

Follow the instructions from [pyTorch website](https://pytorch.org/).

### Install MNE-python

Follow the instructions from [MNE-python website](https://mne.tools/stable/install/index.html).

### Install package from source

```bash
git clone https://github.com/PythonOT/convolutional-monge-mapping-normalization.git
cd convolutional-monge-mapping-normalization
```

In a dedicated Python env, run:

```bash

pip install -e .
```

Also install the requirements:

```bash
pip install -r requirements.txt
```

you might need to install torch with proper cuda version before the requirements above depending on your machine.


### To run the example of the paper for physionet data

```bash
python examples/physionet_experiment.py
```

### To visualize cmmn effect on the data
    
```bash
python examples/cmmn_visualization.py
```
