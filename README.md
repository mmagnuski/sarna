# sarna

<img src="http://zhr59sarny.blox.pl/resource/sarna1.jpg" width="300px">

## Installation
Installing `sarna` requires `git` to be installed.
Recommended way to install `sarna` (from github) is to:
```
pip install git+https://github.com/mmagnuski/sarna
```

### advanced installation
If you know what you are doing and want to have *editable, 'develop'* version of `sarna`, you can install either this way:
```
pip install -e git+https://github.com/mmagnuski/sarna#egg=sarna
```
or this way:
```
git clone https://github.com/mmagnuski/sarna.git
cd sarna
python setup.py develop
```

## Dependencies
You need to have these packages:
* `numpy`
* `matplotlib`
* `pandas`
* [`mne`](https://martinos.org/mne/stable/index.html)
* [`borsar`](https://github.com/mmagnuski/borsar)

Additionally, parts of the code may rely on:
* `seaborn`
