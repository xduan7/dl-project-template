# Deep Learning Project Template
This template offers a lightweight yet functional project template for various deep learning projects. 
The template assumes [PyTorch](https://pytorch.org/) as the deep learning framework.
However, one can easily transfer and utilize the template to any project implemented with other frameworks.


## Table of Contents  
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Template Layout](#template-layout)
- [Extra Packages](#extra-packages)
    - [Data Validation and Cleaning](#data-validation-and-cleaning)
    - [Performance and Caching](#performance-and-caching)
    - [Data Version Control and Workflow](#data-version-control-and-workflow)
    - [Visualization and Analysis](#visualization-and-analysis)
    - [Project Lifecycles and Hyperparameter Optimization](#project-lifecycles-and-hyperparameter-optimization)
    - [PyTorch Extensions](#pytorch-extensions)
    - [Miscellaneous](#miscellaneous)
- [Resources](#resources)
    - [Datasets](#datasets)
    - [Readings](#readings)
    - [Other ML/DL Templates](#other-mldl-templates)
- [Future Tasks](#future-tasks)
- [Authors](#authors)
- [License](#license)


## Dependencies
```text
required:
    python>=3.7
    numpy>=1.18
    pandas>=1.0
    torch>=1.4
    scikit-learn>=0.22

optional:
    poetry>=0.12
    flake8>=3.7
    pylint>=2.4
    mypy>=0.76
    pytest>=5.3
    GPUtil>=1.4
```


## Getting Started
You can fork this repo and use it as a template when creating a new repo on Github like this:
<p align="center">
    <img src="https://github.com/xduan7/dl-project-template/blob/master/docs/readme/create_a_new_repo.png" width="90%">
</p>
Or directly use the template from the forked template repo like this:
<p align="center">
    <img src="https://github.com/xduan7/dl-project-template/blob/master/docs/readme/use_template.png" width="90%">
</p>

Alternatively, you can simply download this repo in zipped format and get started:
<p align="center">
    <img src="https://github.com/xduan7/dl-project-template/blob/master/docs/readme/download.png" width="90%">
</p>

Next, you can install all the dependencies by typing the following command in project root:
```bash
make install # or "poetry install"
```

Finally, you can wrap up the setup by manually install and update any packages you'd like. 
Please refer to the [Extra Packages](#extra-packages) section for some awesome packages. 


## Template Layout
The project layout with the usage for each folder is shown below:
```text
dl-project-template
.
├── LICENSE.md
├── README.md
├── makefile            # makefile for various commands (install, train, pytest, mypy, lint, etc.) 
├── mypy.ini            # MyPy type checking configurations
├── pylint.rc           # Pylint code quality checking configurations
├── pyproject.toml      # Poetry project and environment configurations
|
├── data
|   ├── ...             # data reference files (index, readme, etc.)
│   ├── raw             # untreated data directly downloaded from source
│   ├── interim         # intermediate data processing results
│   └── processed       # processed data (features and targets) ready for learning
├── notebooks           # Jupyter Notebooks (mostly for data processing and visualization)
│── src    
│   │── ...             # top-level scripts for training, testing and downloading
│   ├── processes       # data processing functions and classes (cleaning, validation, imputation etc.)
│   ├── modules         # activations, layers, modules, and networks (subclass of torch.nn.Module)
│   ├── optimization    # deep learning optimizers and schedulers
│   ├── configs         # configuration files for deep learning experiments
│   └── utilities       # other useful functions and classes
├── tests               # unit tests module for ./src
│
├── docs                # documentations (*.txt, *.doc, *.jpeg, etc.)
├── logs                # logs for deep learning experiments
└── models              # saved models with optimizer states
```


## Extra Packages
### Data Validation and Cleaning
- [Great Expectation](https://docs.greatexpectations.io/en/latest/): data validation, documenting, and profiling
- [Cerberus](http://docs.python-cerberus.org/en/stable/): lightweight data validation functionality
- [PyJanitor](https://pyjanitor.readthedocs.io/): Pandas extension for data cleaning
- [PyDQC](https://github.com/SauceCat/pydqc): automatic data quality checking

### Performance and Caching
- [Numba](https://numba.pydata.org/): JIT compiler that translates Python and NumPy to fast machine code
- [Dask](https://dask.org/): parallel computing library
- [Ray](https://ray.io/): framework for distributed applications
- [Modin](http://modin.readthedocs.io/): parallelized Pandas with [Dask](https://dask.org/) or [Ray](https://ray.io/)
- [Vaex](https://vaex.readthedocs.io/en/latest/index.html): lazy memory-mapping dataframe for big data
- [Joblib](https://joblib.readthedocs.io/en/latest/): disk-caching and parallelization
- [RAPIDS](https://rapids.ai/): GPU acceleration for data science

### Data Version Control and Workflow
- [DVC](https://dvc.org/): data version control system
- [Pachyderm](https://www.pachyderm.com/): data pipelining (versioning, lineage/tracking, and parallelization)
- [d6tflow](https://d6tflow.readthedocs.io/en/latest/): effective data workflow
- [Metaflow](https://metaflow.org/): end-to-end independent workflow 

### Visualization and Analysis
- [Seaborn](https://seaborn.pydata.org/): data visualization based on [Matplotlib](https://matplotlib.org/)
- [HiPlot](https://facebookresearch.github.io/hiplot/): interactive high-dimensional visualization for correlation and pattern discovery
- [Plotly.py](https://plot.ly/python/): interactive browser-based graphing library
- [Altair](https://altair-viz.github.io/): declarative visualization based on [Vega](http://vega.github.io/vega) and [Vega-Lite](http://vega.github.io/vega-lite)
- [TabPy](https://tableau.github.io/TabPy/docs/about.html): [Tableau](https://www.tableau.com/) visualizations with Python
- [Chartify](https://github.com/spotify/chartify): easy and flexible charts
- [Pandas-Profiling](https://pandas-profiling.github.io/pandas-profiling/docs/): HTML profiling reports for Pandas DataFrames

### Project Lifecycles and Hyperparameter Optimization
- [NNI](https://nni.readthedocs.io/en/latest/): automate ML/DL lifecycle (feature engineering, neural architecture search, model compression and hyperparameter tuning)
- [Comet.ml](https://www.comet.ml/site/): self-hosted and cloud-based meta machine learning platform for tracking, comparing, explaining and optimizing experiments and models
- [MLflow](https://mlflow.org/): platform for ML lifecycle , including experimentation, reproducibility and deployment
- [Optuna](https://optuna.org/): automatic hyperparameter optimization framework
- [Hyperopt](http://hyperopt.github.io/hyperopt): serial and parallel optimization
- [Tune](https://ray.readthedocs.io/en/latest/tune.html): scalable experiment execution and hyperparameter tuning

### PyTorch Extensions
- [Ignite](https://pytorch.org/ignite/): high-level library based on PyTorch
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning): lightweight wrapper for less boilerplate
- [RaySGD](https://ray.readthedocs.io/en/latest/raysgd/raysgd.html): lightweight wrappers for distributed deep learning
- [fastai](https://docs.fast.ai/): out-of-the-box tools and models for vision, text, and other data
- [Skorch](https://skorch.readthedocs.io/en/latest/?badge=latest): [Scikit-Learn](https://scikit-learn.org/stable/index.html) interface for PyTorch models
- [PyRo](https://pyro.ai/): deep universal probabilistic programming with PyTorch
- [Kornia](https://kornia.org/): differentiable computer vision library
- [DGL](http://dgl.ai/): package for deep learning on graphs
- [PyGeometric](https://pytorch-geometric.readthedocs.io/): geometric deep learning extension library for PyTorch
- [Torchmeta](https://tristandeleu.github.io/pytorch-meta/): datasets and models for few-shot-learning/meta-learning
- [learn2learn](http://learn2learn.net/): meta-learning model implementations
- [higher](https://higher.readthedocs.io/en/latest/): higher-order (unrolled first-order) optimization
- [Captum](https://captum.ai/): model interpretability and understanding
- [PyTorch summary](https://github.com/sksq96/pytorch-summary): Keras style summary for PyTorch models

### Miscellaneous
- [DoWhy](https://microsoft.github.io/dowhy/): causal inference combining causal graphical models and potential outcomes
- [NetworkX](https://networkx.github.io/documentation/stable/): creation, manipulation, and study of complex networks/graphs
- [Gym](https://gym.openai.com/): toolkit for developing and comparing reinforcement learning algorithms


## Resources
### Datasets:
- [Google Datasets](https://cloud.google.com/public-datasets): high-demand public datasets
- [Google Dataset Search](https://datasetsearch.research.google.com): a search engine for freely-available online data
- [OpenML](https://www.openml.org/): online platform for sharing data, ML algorithms and experiments
- [OpenBlender](https://www.openblender.io): live-streamed open data sources

### Readings:
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design) by Chip Huyen
- [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) by Martin Zinkevich
 
### Other ML/DL Templates:
- [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science): a logical, reasonably standardized, but flexible project structure
- [PyTorch Template Project](https://github.com/victoresque/pytorch-template): PyTorch deep learning project template


## Future Tasks
- [ ] ML/DL projects process flowchart 
    - [ ] definition of several major steps
    - [ ] clarify motivation and deliverables
- [ ] small example for demonstration (omniglot?)


## Authors
* Xiaotian Duan (Email: xduan7 at uchicago.edu)


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for more details.

