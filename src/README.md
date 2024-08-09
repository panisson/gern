Please put your code in this directory.
It is a good idea to use Python files instead of notebooks for code that is meant to be executed many times: for instance, an algorithm or a ML model under development.

All code dependencies (Python packages) should be specified by the conda environment described in `../environment.yml`.
You can create this conda environment with this command:

    conda env create --file ../environment.yml

If you modify the `environment.yml` file, then you can update the (existing) conda enviroment by using:

    conda env update --file ../environment.yml --prune
