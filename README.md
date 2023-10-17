# Convolutional Monge Mapping Normalization for learning on Sleep data

This repository is the official implementation of Convolutional Monge Mapping Normalization for learning on Sleep data. This repository proposes an example of the method applied to Physionet data available on [MNE-python](https://mne.tools/stable/index.html). 

(The other datasets used in the paper are available on request. That the reason why we propose an example on Physionet data only.)

### Install pyTorch

Follow the instructions from [pyTorch website](https://pytorch.org/).

### Install MNE-python

Follow the instructions from [MNE-python website](https://mne.tools/stable/install/index.html).

### Install package from source

```bash
git clone https://github.com/tgnassou/cmmn.git
cd cmmn
```

In a dedicated Python env, run:

```bash
pip install -e .
```

### To run the example of the paper for physionet data

```bash
python examples/physionet.py
```

### To visualize cmmn effect on the data
    
```bash
python examples/cmmn_visualization.py
```

### To run faster experiments

```bash
python examples/fast_example.py --fast
```
