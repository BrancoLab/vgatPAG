# vgatPAG
Note that this code needs an older version of `fcutils`: https://github.com/FedeClaudi/fcutils/tree/06703a18bcfa6b473ba5f4a80803fdd89e5a52b2

Note that you'll need `datajoint=0.12.4`.

## Setup for Caiman
Currently working of the dev branch of caiman, install with these steps:

Clone repository and switch to dev branch

```
git clone https://github.com/flatironinstitute/CaImAn.git
cd caiman
git checkout -b 1p_rt
git pull
```

Crate a conda environemnt install dependencies and then caiman
```
conda create -n cai python==3.6.4
conda activate cai
pip install -r requirements.txt
pip install -U ipykernel
pip install pyqt5 napari fancylog
pip install . --user
```


## Other packages for behaviour analysis

Requires Behaviour and fcutils packages [written by Federico and available on github], install with:

```
pip install git+https://github.com/BrancoLab/Behaviour.git --upgrade
```

```
pip install git+https://github.com/FedeClaudi/fcutils.git --upgrade
```