Hello! This project is organized with a typical data science project structure:

- `notebook` contains Jupyter notebooks: code that has been run at a specific point in time.
- `data` contains all (raw and processed) data.
- `paper` contains all **figures**, documents, presentations, related to this project.
- `src` contains source files (e.g. `*.py`): code that is meant to be executed multiple times, in multiple notebooks.

Each of these directories contains a small README with further information on how the project is organized. It helps reproducibility if all Python code in this project is run inside the [Conda](https://docs.conda.io/en/latest/) environment specified by `environment.yml`.

Feel free to modify the `.gitignore` files in order for `git` to include only the files that should actually be in the repository (so that e.g. `git add .` always add only the file you actually want).



To install the environment and use the Jupyter notebooks, run (assuming you already have `conda` and `ipykernel` installed):

    conda env create -f environment.yml
    python -m ipykernel install --user --name=pyg

To update the environment:

    conda env update -f environment.yml --prune