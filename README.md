# sarna

<img src="https://www.naturnik.pl/wp-content/uploads/2017/01/sarna-po%C5%A4ciel-www-500x500.jpg" width="300px">

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
