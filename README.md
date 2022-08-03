# JupyterReviewer

A package for using and creating interactive dashboards for manual review.

![Purity Jupyter Reviewer](https://github.com/getzlab/JupyterReviewer/blob/master/images/ezgif.com-gif-maker.gif)

## Quick Start

### Install

1. Download the repository: `git clone git@github.com:getzlab/JupyterReviewer.git` 
1. `cd JupyterReviewer`
1. Create an environment: `conda create --name <my-env> --file requirements.txt`
1. Install package: `pip install -e .`

### Start reviewing

1. Open or create a new jupyter notebook
1. Pick a reviewer (see `JupyterReviewer.Reviewers`)
   ```
   from JupyterReviewer.Reviewers import ExampleReviewer
   ```
1. Instantiate the reviewer
   ```
   my_example_reviewer = ExampleReviewer()
   ```
1. Set the review data*
   ```
   my_example_reviewer.set_review_data(...)
   ```
1. Set the app*
   ```
   my_example_reviewer.set_review_app(...)
   ```
1. Set default annotation configurations
   ```
   my_example_reviewer.set_default_review_data_annotations_configuration(...)
   ```
1. Set default autofill configurations
   ```
   my_example_reviewer.set_default_autofill(...)
   ```
1. Run the reviewer!
   ```
   my_example_reviewer.run()
   ```

*In jupyter notebook, place the cursor at the end of the function call and press `Shift+Tab` to view the docstring.

See a more detailed tutorial in `example_notebooks/Intro_to_Reviewers.ipynb`

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

The `JupyterReviewer` package makes it simple to create dashboards for reviewing data. Developers and users can easily customize their dashboards to incorpate any data they like, and automatically provides a reviewer an easy way to annotate their data, track changes, and share their annotations with others.

### Get Started

See `example_notebooks/` for documentation and tutorials.