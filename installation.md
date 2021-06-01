# Local Installation

Follow these steps to run the notebook on a local
Jupyter installation (this will offer you the full
interactive functionality).

## Installation

```
cd /path/to/your/projects/folder
git clone https://github.com/poke1024/vectorian-2021-demo
cd vectorian-2021-demo
conda env create -f environment.yml
```

The `conda` command usually takes 5 to 10 minutes.

## Launching

```
conda activate vectorian-demo
cd /path/to/your/projects/folder
cd vectorian-2021-demo
jupyter notebook publication.ipynb
```

Executing the first code cell will take some time.

# Via Binder

Go to `https://mybinder.org/` and select the repository
and the correct branch. Select the `publication.ipynb`
as notebook, then click "launch".
