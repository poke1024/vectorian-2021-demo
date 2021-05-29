# vectorian-2021-demo
Demo Notebooks for the new Vectorian

# Local Installation

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

## Troubleshooting

Sometimes the Bokeh server wants a different port. Running

```
import os
os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "localhost:8890"
```

inside Jupyter can help.
