# Deep Learning Project Template
This template offers a lightweight yet functional project template for various deep learning projects. 
The template assumes [PyTorch](https://pytorch.org/) as the deep learning framework.
However, one can easily transfer and utilize the template to any project implemented with other frameworks.


<!---
a table of content section might be a good idea
-->

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
    pylint>=2.4
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
[dl-project-template] tree .                                                                           
.
├── ...                 # project config (requirements, license, etc.)
├── data
|   ├── ...             # data indexes and other descriptor files
│   ├── raw             # downloaded and untreated data
│   ├── cached          # cached files during processing
│   ├── interm          # intermediate results during processing
│   └── processed       # processed data (features, targets) ready for learning
├── docs                # documentations (txt, doc, jpeg, etc.)
├── logs                # logs generated from programs
├── models              # saved model parameters with optimizer
├── notebooks           # jupyter notebooks for experiments and visualization 
├── src    
│   ├── ...             # top-level scripts for deep learning
│   ├── configs         # configurations (*.py) for deep learning experiments
│   ├── processes       # data processing functions and classes
│   ├── modules         # layers, modules, and networks
│   ├── optimization    # optimizers and schedulers
│   └── utilities       # other useful functions and classes
└── tests               # tests for data processing and learning loops
```


<!---
## Feature Usage
use cases for implemented features
-->


## Future Tasks
- [x] python environment setup (pyproject.toml, Makefile, etc.)
- [x] commonly-used utility functions
    - [x] random seeding (for Numpy, PyTorch, etc.)
    - [x] gpu/cpu specification
    - [x] debug decorator
- [ ] customizable neural network modules
    - [ ] fully-connected block
    - [ ] ResNet block
    - [ ] attention block
- [ ] customizable optimization functions
    - [ ] optimizer constructor
    - [ ] learning rate scheduler constructor 
- [x] minimal setup for hyperparameter optimization
    - [x] configuration file
- [ ] process flowchart for DL/ML projects
- [x] extra packages for DL/ML projects
- [ ] documentation
    - [ ] getting started
    - [ ] module docstrings


## Extra Packages
### Data Validation and Cleaning
- [Great Expectation](https://docs.greatexpectations.io/en/latest/): data **validation**, **documenting**, and **profiling**
- [PyJanitor](https://pyjanitor.readthedocs.io/): Pandas extension for data **cleaning**
- [PyDQC](https://github.com/SauceCat/pydqc): automatic data **quality checking**

Performance and Caching
- [Numba](https://numba.pydata.org/): JIT compiler that translates Python and NumPy to fast machine code
- [Dask](https://dask.org/): parallel computing library
- [Ray](https://ray.io/): framework for distributed applications
- [Modin](http://modin.readthedocs.io/): parallelized Pandas with [Dask](https://dask.org/) or [Ray](https://ray.io/) 
- [Joblib](https://joblib.readthedocs.io/en/latest/): disk-caching and parallelization

Version Control and Workflows
- [DVC](https://dvc.org/): a open source version control system for machine learning projects
- [Pachyderm](https://www.pachyderm.com/): a tool for production data pipelines (versioning, lineage, and parallelization)
- [d6tflow](https://d6tflow.readthedocs.io/en/latest/): a Python library for building highly effective data science workflows

Visualization and Analysis
- [Seaborn](https://seaborn.pydata.org/): a Python data visualization library based on matplotlib
- [Plotly.py](https://plot.ly/python/): an open-source, interactive graphing library for Python
- [Altair](https://altair-viz.github.io/): a declarative statistical visualization library for Python, based on Vega and Vega-Lite
- [Chartify](https://github.com/spotify/chartify): a Python library that makes it easy for data scientists to create charts
- [Pandas-Profiling](https://pandas-profiling.github.io/pandas-profiling/docs/): create HTML profiling reports from pandas DataFrame objects

Machine Learning Lifecycles
- [NNI](https://nni.readthedocs.io/en/latest/): a lightweight but powerful toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyperparameter tuning
- [Comet.ml](https://www.comet.ml/site/): a self-hosted and cloud-based meta machine learning platform allowing data scientists and teams to track, compare, explain and optimize experiments and models

Hyperparameter Optimization
- [Optuna](https://optuna.org/): an automatic hyperparameter optimization software framework
- [Hyperopt](http://hyperopt.github.io/hyperopt): a Python library for serial and parallel optimization over awkward search spaces
- [MLflow](https://mlflow.org/): an open source platform to manage the ML lifecycle, including experimentation, reproducibility and deployment

PyTorch Extensions
- [Ignite](https://pytorch.org/ignite/): is a high-level library to help with training neural networks in PyTorch
- [PyRo](https://pyro.ai/): deep universal probabilistic programming with Python and PyTorch
- [DGL](http://dgl.ai/): an easy-to-use, high performance and scalable Python package for deep learning on graphs
- [PyGeometric](https://pytorch-geometric.readthedocs.io/): a geometric deep learning extension library for PyTorch


## Resources
Readings:
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design) by Chip Huyen
- [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) by Martin Zinkevich
 
Other Machine Learning Template Repos:
- [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science): "a logical, reasonably standardized, but flexible project structure for doing and sharing data science work"
- [PyTorch Template Project](https://github.com/victoresque/pytorch-template): "PyTorch deep learning project made easy"
- [PyTorch Project Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template): "a best practice for deep learning project template architecture"

Datasets:
- [Google Dataset Search](https://datasetsearch.research.google.com): "a search engine from Google that helps researchers locate online data that is freely available for use"
- [OpenML](https://www.openml.org/): "an online machine learning platform for sharing and organizing data, machine learning algorithms and experiments"
- [OpenBlender](https://www.openblender.io): "self-service platform to enrich datasets with correlated variables from thousands of live-streamed open data sources"


## Authors
* Xiaotian Duan (Email: xduan7 at uchicago.edu)


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for more details.

