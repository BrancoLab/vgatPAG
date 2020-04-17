# vgatPAG


## Setup for Caiman
Currently working of the dev branch of caiman, install with these steps:

Clone repository and switch to dev branch

```
git clonehttps://github.com/flatironinstitute/CaImAn.git
cd caiman
git checkout dev
git pull
```

Crate a conda environemnt install dependencies and then caiman
```
conda create -n cai python==3.6.4
conda activate cai
pip install -r requirements.txt
pip install -U ipykernel
pip install pyqt5 napari fancylog
pip install .
```

## Other packages
Other packages in processing of calcium data can be installed with:
```
pip install napari
```


## Other packages for behaviour analysis

Requires Behaviour and fcutils packages [written by Federico and available on github], install with:

```
pip install git+https://github.com/BrancoLab/Behaviour.git --upgrade
```

```
pip install git+https://github.com/FedeClaudi/fcutils.git --upgrade
```