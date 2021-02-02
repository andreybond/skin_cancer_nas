skin_cancer_nas
==============================

Skin cancer NAS project. Deep Learning applied to 440-950 nm waveband photo images of skin lesions classification.

This is second phase of the project (first one is 'skin_cancer') and it is related to AutoKeras (NAS) (Unsuccessfull) and NNI DARTS utilization for classifier buildup using 3+1 channels.

Outcome - we have been able to search for the 1-5 meta-metalayer deep architectures trained on 128x128 pixels images with slightly reduced original search space (convolutions of size 3x3, 5x5 only).

The examples of the found architectures can be found in PDF files /notebooks/4_DARTS/:

 - Found 1-meta-layer architecture: <a href="https://gitlab.com/rtu_lu_rnd/skin_cancer_nas/-/tree/master/notebooks/4_DARTS/1_layer.pdf" target="_blank">1_layer.pdf</a>
 - Found 2-meta-layer architecture: <a href="https://gitlab.com/rtu_lu_rnd/skin_cancer_nas/-/tree/master/notebooks/4_DARTS/2_layer.pdf" target="_blank">2_layer.pdf</a>
 - Found 3-meta-layer architecture: <a href="https://gitlab.com/rtu_lu_rnd/skin_cancer_nas/-/tree/master/notebooks/4_DARTS/3_layer.pdf" target="_blank">3_layer.pdf</a>
 - Found 4-meta-layer architecture: <a href="https://gitlab.com/rtu_lu_rnd/skin_cancer_nas/-/tree/master/notebooks/4_DARTS/4_layer.pdf" target="_blank">4_layer.pdf</a>
 - Found 5-meta-layer architecture: <a href="https://gitlab.com/rtu_lu_rnd/skin_cancer_nas/-/tree/master/notebooks/4_DARTS/5_layer.pdf" target="_blank">5_layer.pdf</a>

Project Organization (Cookiecutter template)
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
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
