Master's Thesis
==============================

## Thesis Title: [Geogenic radon mapping of Hessen using Machine Learning Techniques](https://github.com/Madaar49/Masters_Thesis/tree/main)

## Overview

This research focused on spatial modelling of *Geogenic radonpotential (GRP)* in Hessen District using machine learning techniques, and environmental covariables. 

*Geogenic radonpotential (GRP)* is defined as the portion of radon emanation, that is predominantly associated with natural factors.

**Hypothesis :** It is hypothesizes that the spatial variability of GRP is affected by environmental parameters related to soil, geology, meterology etc. Therefore, by modelling a relationship between known sampling points and their environmental co-variables, we can predict the radon potential in areas where the environmental co-variables are present but radon is unknown.


## Data and Method
- geogenic radonpotential sampling data.

- 38 covariates related to geology, soil, climate, Uranium, DEM etc.

- Use spatial cross validation for feature selection.

-  Develop models such as `Random Forest`, `xGBoost`, `Suport Vector Regressor`, `Multi-Layer-Perceptron Regressor`

## Deliverables

The code and workflow will be available after here ater publication of the work.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
