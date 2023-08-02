# AnnoMate

A package for using and creating interactive dashboards for manual review.

![Purity AnnoMate Reviewer](https://github.com/getzlab/AnnoMate/blob/master/images/ezgif.com-gif-maker.gif)

# Quick Start

## Install

### Set up Conda Environment

This is _highly_ recommended to manage different dependencies required by different reviewers.

1. Install conda

    Credit to Raymond Chu this article: https://medium.com/google-cloud/set-up-anaconda-under-google-cloud-vm-on-windows-f71fc1064bd7

    ```
    sudo apt-get update
    sudo apt-get install bzip2 libxml2-dev

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    rm Miniconda3-latest-Linux-x86_64.sh
    source .bashrc
    conda install scikit-learn pandas jupyter ipython
    ```

2. Create a conda environment

    If you do not already have a designated environment: 
    
    ```
    conda create --name <your_env> python==<py_version>
    ```

    `<your_env>` is the name of your environment (ie purity_review_env). Check the corresponding reviewer's `setup.py` file to get the proper python version for `py_version`.

3. Add conda environment to ipykernel 

    Credit to Nihal Sangeeth from StackOverflow: https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab.

    ```
    conda activate <your_env>
    conda install ipykernel
    ipython kernel install --user --name=<your_env>
    conda deactivate
    ```

    When you open a jupyter notebook, you can change the environment the notebook cells are run in to `<your_env>`


### Install AnnoMate with pip

If you are developing a brand new reviewer, you can install from PyPi

```
conda activate <your_env>
pip install AnnoMate
```

### Install with Git

AnnoMate and most prebuilt reviewers can be downloaded with git. 

```
git clone git@github.com:getzlab/AnnoMate.git
cd AnnoMate
conda activate <your_env> --file requirements.txt
pip install -e .
```

### Tutorials and Documentation

See a more detailed tutorial in `tutorial_notebooks/Intro_to_Reviewers.ipynb`.

View the catalog of existing reviewers at [catalog/ReviewerCatalog.ipynb](https://github.com/getzlab/AnnoMate/blob/master/catalog/ReviewerCatalog.ipynb).

For developers, see `tutorial_notebooks/Developer_Jupyter_Reviewer_Tutorial.ipynb`.

## Why Jupyter Reviewer
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
