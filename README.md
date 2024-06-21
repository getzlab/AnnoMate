# AnnoMate

A package for using and creating interactive dashboards for manual review.

## Quick Start

You can run the Quick Example notebook on Google Colab: [Quick_Example_AnnoMate_Reviewer.ipynb](https://colab.research.google.com/github/getzlab/AnnoMate/blob/master/tutorial_notebooks/Quick_Example_AnnoMate_Reviewer.ipynb)
- Open the link, which gives Viewer access.
- Next to the title of notebook it will say `Changes will not be saved`. Click the link to make a copy that is saved to your google drive.
- Follow the Google Colab instructions to install AnnoMate in your Colab instance.
- Work through the tutorial to understand how to use a pre-built Reviewer and make simple modifications within a notebook.

## Install

### Set up Conda Environment

This is _highly_ recommended to manage different dependencies required by different reviewers.

1. Install conda if you do not have it already<sup>1</sup>

2. Create a conda environment

    If you do not already have a designated environment: 
    
    ```
    conda create --name <your_env> python==<py_version>
    ```
    
    `<your_env>` is the name of your environment (ie purity_review_env). Check the corresponding reviewer's `setup.py` file to get the proper python version for `py_version`. Reviewers have been tested on `3.8` and `3.9`.
   
4. Activate your conda environment
   ```
   conda activate <your_env>
   ```
   **You'll want your environment activated when you install AnnoMate in any of the below mentioned ways** 

5. Add conda environment to ipykernel<sup>2</sup>

    When you open a jupyter notebook, you can change the environment the notebook cells are run in to `<your_env>`
    
#### Option 1: Install AnnoMate with pip

If you are developing a brand new reviewer, you can install from PyPi

```
pip install AnnoMate
```

#### Option 2: Install with Git

AnnoMate and most prebuilt reviewers can be downloaded with git. 

```
git clone git@github.com:getzlab/AnnoMate.git
cd AnnoMate
pip install -e .
```

#### Option 3: Install with Conda

Assuming you already have a conda environment and it is activated:
```
conda env update --name <your_env> --file annomate_conda_environment.yml
```

If you have not made a new conda environment:
```
conda env create --file annomate_conda_environment.yml --name <your_env>
conda activate <your_env>
```
Make sure to add conda environment to ipykernel (see **4. Add conda environment to ipykernel**)

### Run with a Docker container

Below are the commands needed to run the tutorial notebooks in a docker container. You need to open at least 2 ports if you are using Mac or Windows:
1. A port to open jupyter lab (in this case, `<jupyter_port>`, i.e. 8890. We recommend to NOT use port 8888, and make sure to not use other ports in use)
2. A port to open the dash app (`<dash_port>`)
```
docker run -it -p <jupyter_port>:<jupyter_port> -p <dash_port>:<dash_port> ghcr.io/getzlab/annomate:latest
cd AnnoMate/tutorial_notebooks
jupyter lab --ip 0.0.0.0 --port <jupyter_port> --no-browser --allow-root
```
Copy the provided link to a browser to open and run the jupyter notebooks. In the notebooks, you can set the port to open the Dash app in `reviewer.run(port=<dash_port>, ...)`.

If you are running on Linux, you can open all ports.
```
docker run -it --network host ghcr.io/getzlab/annomate:latest
cd AnnoMate/tutorial_notebooks
jupyter lab --ip 0.0.0.0 --port <jupyter_port> --no-browser --allow-root
```
Make sure to try both available links. In the past we have been able to open the notebook using the `http://127.0.0.1` link.

## Tutorials and Documentation

To run tutorial notebooks, clone the repository if you have not already. If you are running from the docker container, the repo has already been cloned.

```
git clone git@github.com:getzlab/AnnoMate.git
cd AnnoMate/tutorial_notebooks
```

Make sure to set the notebook kernels to the environment with the `AnnoMate` package.

- See a more detailed tutorial in `tutorial_notebooks/Intro_to_Reviewers.ipynb`. 
- View the catalog of existing reviewers at [catalog/ReviewerCatalog.ipynb](https://github.com/getzlab/AnnoMate/blob/master/catalog/ReviewerCatalog.ipynb).
- For developers, see `tutorial_notebooks/Developer_Jupyter_Reviewer_Tutorial.ipynb`.

## Why AnnoMate
### Why and how we review data

Part of any study is ensuring data are consistent and drawing conclusions about the data from multiple sources. Studies are often novel, so frequently there are steps along the way that do not have existing, validated automation techniques. Therefore, we must perform manual review.

Typically, the person reviewing all this data opens a bunch of windows to view data from different places (a clinical information spreadsheet from a collaborator, a few outputs from a Terra workflow, and/or previous notes from another reviewer, etc.). Next they look at all the data and keep notes in yet a separate document, such as a spreadsheet or digital/physical notes. Then, they go row by row, sample by sample, until they finish.

### Why we need something better

While straightforward to do in theory, this review method is very brittle, error prone, and very time consuming. 

Reviewing can take a very long time, such as reviewing large datasets on the order of hundreds to thousands of data points, or if the review needs to be repeated multiple times due to changes in processes upstream. 

Some review processes are iterative, or new information is gained from some other source to inform the review process, or we need to pass off the review process to someone else. We should be able to easily incorporate old data with new data, and share that history and information with others.

Some reviews require calculations, or exploring the the data in ways that a static plot cannot provide. Some Terra workflows do produce some interactive html files, but this is rare. Sometimes, a reviewer realizes mid-way through the review process that a different kind of plot could be very informative. It should be easy to generate such a plot on the fly without having to modify or create a new Terra workflow, or opening a new notebook to calculate manually.

Lastly, humans are humans, and we make mistakes. It can be very tedious to maintain and update a large spreadsheet with hundreds of rows and multiple columns to annotate. Annotations are difficult to enforce in this setting, and changes (intentional or accidental) are difficult to track. 

### The Solution: Jupyter notebook and Plotly-Dash!

Most ACBs use jupyter notebooks for their analysis. So why not keep the review process in jupyter notebooks too? Additionally, there already exist great tools for making interactive figures and dashboards. We can use these packages to help automatically consildate information and create figures that will make it easier to review, enforce annotation standards, and track changes over time.

The `AnnoMate` package makes it simple to create dashboards for reviewing data. Developers and users can easily customize their dashboards to incorpate any data they like, and automatically provides a reviewer an easy way to annotate their data, track changes, and share their annotations with others.

### Get Started

See `tutorial_notebooks/` for documentation and tutorials.

# For AnnoMate Developers

New features to AnnoMate are pushed to `dev_branch`. Any separate large features being developed simultaneously can be done in separate branches, and merged to `dev_branch` first. 

Only when we merge the `dev_branch` to master, we also push to pypi. At this time we decide the new version number.

Semantic versioning (https://packaging.python.org/en/latest/discussions/versioning/)
The idea of semantic versioning (or SemVer) is to use 3-part version numbers, major.minor.patch, where the project author increments:
- major when they make incompatible API changes,
- minor when they add functionality in a backwards-compatible manner, and
- patch, when they make backwards-compatible bug fixes.

For AnnoMate:
- Major: Not backwards compatible
- Minor: up to 3 functionality changes with backwards compatibility within a year
    - pickle versioning in reviewdatainterface
    - hot keys
    - Adding components or new custom pre-built reviewers 
- Patch: up to 5 bug fixes within 6 months

Github continuous integration: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python

# Supplements 

1. Credit to Raymond Chu this article: https://medium.com/google-cloud/set-up-anaconda-under-google-cloud-vm-on-windows-f71fc1064bd7

    ```
    sudo apt-get update
    sudo apt-get install bzip2 libxml2-dev

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    rm Miniconda3-latest-Linux-x86_64.sh
    source .bashrc
    conda install scikit-learn pandas jupyter ipython
    ```
    

2. Credit to Nihal Sangeeth from StackOverflow: https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab.

    ```
    conda install ipykernel
    ipython kernel install --user --name=<your_env>
    conda deactivate
    ```
