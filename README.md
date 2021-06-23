# Deep Learning Project Template
This template offers a lightweight yet functional project template for various deep learning projects. 
The template assumes [PyTorch](https://pytorch.org/) as the deep learning framework.
However, one can easily transfer and utilize the template to any project implemented with other frameworks.


## Table of Contents
- [Getting Started](#getting-started)
- [Template Layout](#template-layout)
- [Extra Packages](#extra-packages)
    - [Data Analysis, Validation, and Cleaning](#data-analysis-validation-and-cleaning)
    - [Performance and Caching](#performance-and-caching)
    - [Data Version Control and Workflow](#data-version-control-and-workflow)
    - [Visualization and Presentation](#visualization-and-presentation)
    - [Project Lifecycles and Hyperparameter Optimization](#project-lifecycles-and-hyperparameter-optimization)
    - [Other PyTorch Extensions](#other-pytorch-extensions)
    - [Miscellaneous](#miscellaneous)
- [Resources](#resources)
    - [Datasets](#datasets)
    - [Readings](#readings)
    - [Other ML/DL Templates](#other-mldl-templates)
- [Authors](#authors)
- [License](#license)


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
conda careate -n project_name python=3.8
conda install poetry  # or 'pip install poetry'
poetry new project_name
```

Finally, you can wrap up the setup by manually install and update any packages you'd like. 
Please refer to the [Extra Packages](#extra-packages) section for some awesome packages. 


## Template Layout

```text
dl-project-template
.
|
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
|
├── notebooks           # Jupyter Notebooks (mostly for data processing and visualization)
│── src    
│   ├── data            # data processing classes, functions, and scripts
│   ├── evaluations     # evaluation classes and functions (metrics, visualization, etc.)
│   ├── experiments     # experiment configuration files
│   ├── modules         # activations, layers, modules, and networks (subclass of torch.nn.Module)
│   └── utilities       # other useful functions and classes
├── tests               # unit tests module for ./src
│
├── docs                # documentation files (*.txt, *.doc, *.jpeg, etc.)
├── logs                # logs for deep learning experiments
└── models              # saved models with optimizer states
```


## Extra Packages
### Data Analysis, Augmentation, Validation and Cleaning
- [Great Expectation](https://docs.greatexpectations.io/en/latest/): data validation, documenting, and profiling
- [Cerberus](http://docs.python-cerberus.org/en/stable/): lightweight data validation functionality
- [PyJanitor](https://pyjanitor.readthedocs.io/): Pandas extension for data cleaning
- [PyDQC](https://github.com/SauceCat/pydqc): automatic data quality checking
- [Feature-engine](https://feature-engine.readthedocs.io/en/latest/index.html): transformer library for feature preparation and engineering
- [pydantic](https://pydantic-docs.helpmanual.io/): data parsing and validation using Python type hints 
- [Dora](https://github.com/NathanEpstein/Dora): exploratory data analysis toolkit for Python
- [datacleaner](https://github.com/rhiever/datacleaner): automatically cleans data sets and readies them for analysis
- [whale](https://docs.whale.cx/): a lightweight data discovery, documentation, and quality engine for data warehouse
- [bamboolib](https://bamboolib.8080labs.com/): a tool for fast and easy data exploration & transformation of pandas DataFrames
- [pandas-summary](https://github.com/mouradmourafiq/pandas-summary): an extension to pandas dataframes describe function
- [AugLy](https://github.com/facebookresearch/AugLy): a data augmentations library for audio, image, text, and video. 

### Performance and Caching
- [Numba](https://numba.pydata.org/): JIT compiler that translates Python and NumPy to fast machine code
- [CuPy](https://cupy.dev): NumPy-like API accelerated with CUDA 
- [Dask](https://dask.org/): parallel computing library
- [Ray](https://ray.io/): framework for distributed applications
- [Modin](http://modin.readthedocs.io/): parallelized Pandas with [Dask](https://dask.org/) or [Ray](https://ray.io/)
- [Vaex](https://vaex.readthedocs.io/en/latest/index.html): lazy memory-mapping dataframe for big data
- [Joblib](https://joblib.readthedocs.io/en/latest/): disk-caching and parallelization
- [RAPIDS](https://rapids.ai/): GPU acceleration for data science
- [Polars](https://github.com/ritchie46/polars): a blazingly fast DataFrames library implemented in Rust & Python

### Data Version Control and Workflow
- [DVC](https://dvc.org/): data version control system
- [Pachyderm](https://www.pachyderm.com/): data pipelining (versioning, lineage/tracking, and parallelization)
- [d6tflow](https://d6tflow.readthedocs.io/en/latest/): effective data workflow
- [Metaflow](https://metaflow.org/): end-to-end independent workflow 
- [Dolt](https://github.com/liquidata-inc/dolt): relational database with version control
- [Airflow](https://airflow.apache.org/): platform to programmatically author, schedule and monitor workflows
- [Luigi](https://luigi.readthedocs.io/en/stable/): dependency resolution, workflow management, visualization, etc.

### Visualization and Presentation
- [Seaborn](https://seaborn.pydata.org/): data visualization based on [Matplotlib](https://matplotlib.org/)
- [HiPlot](https://facebookresearch.github.io/hiplot/): interactive high-dimensional visualization for correlation and pattern discovery
- [Plotly.py](https://plot.ly/python/): interactive browser-based graphing library
- [Altair](https://altair-viz.github.io/): declarative visualization based on [Vega](http://vega.github.io/vega) and [Vega-Lite](http://vega.github.io/vega-lite)
- [TabPy](https://tableau.github.io/TabPy/docs/about.html): [Tableau](https://www.tableau.com/) visualizations with Python
- [Chartify](https://github.com/spotify/chartify): easy and flexible charts
- [Pandas-Profiling](https://pandas-profiling.github.io/pandas-profiling/docs/): HTML profiling reports for Pandas DataFrames
- [missingno](https://github.com/ResidentMario/missingno): toolset of flexible and easy-to-use missing data visualizations and utilities
- [Yellowbrick](https://www.scikit-yb.org/en/latest/): Scikit-Learn visualization for model selection and hyperparameter tuning
- [FlashTorch](https://github.com/MisaOgura/flashtorch): visualization toolkit for neural networks in PyTorch
- [Streamlit](https://www.streamlit.io/): turn data scripts into sharable web apps in minutes
- [python-tabulate](https://github.com/astanin/python-tabulate): pretty-print tabular data in Python, a library and a command-line utility
- [Lux](https://github.com/lux-org/lux): Python API for intelligent visual data discovery 
- [bokeh](https://bokeh.org/): interactive data visualization in the browser, from Python
 
### Project Lifecycles and Hyperparameter Optimization
- [NNI](https://nni.readthedocs.io/en/latest/): automate ML/DL lifecycle (feature engineering, neural architecture search, model compression and hyperparameter tuning)
- [Comet.ml](https://www.comet.ml/site/): self-hosted and cloud-based meta machine learning platform for tracking, comparing, explaining and optimizing experiments and models
- [MLflow](https://mlflow.org/): platform for ML lifecycle , including experimentation, reproducibility and deployment
- [Optuna](https://optuna.org/): automatic hyperparameter optimization framework
- [Hyperopt](http://hyperopt.github.io/hyperopt): serial and parallel optimization
- [Tune](https://ray.readthedocs.io/en/latest/tune.html): scalable experiment execution and hyperparameter tuning
- [Determined](https://determined.ai/): deep learning training platform
- [Aim](https://aimstack.io/): a super-easy way to record, search and compare 1000s of ML training runs 
- [TPOT](https://epistasislab.github.io/tpot/): a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming

### Distribution, Pipelining, and Sharding
- [torchgpipe](https://torchgpipe.readthedocs.io/en/stable/): a scalable pipeline parallelism library, which allows efficient training of large, memory-consuming models
- [PipeDream](https://github.com/msr-fiddle/pipedream): generalized pipeline parallelism for deep neural network training
- [DeepSpeed](https://www.deepspeed.ai/): a deep learning optimization library that makes distributed training easy, efficient, and effective
- [Horovod](https://eng.uber.com/horovod/): a distributed deep learning training framework
- [RaySGD](https://ray.readthedocs.io/en/latest/raysgd/raysgd.html): lightweight wrappers for distributed deep learning
- [AdaptDL](https://adaptdl.readthedocs.io): a resource-adaptive deep learning training and scheduling framework

### Other PyTorch Extensions
- [Ignite](https://pytorch.org/ignite/): high-level library based on PyTorch
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning): lightweight wrapper for less boilerplate
- [fastai](https://docs.fast.ai/): out-of-the-box tools and models for vision, text, and other data
- [Skorch](https://skorch.readthedocs.io/en/latest/?badge=latest): [Scikit-Learn](https://scikit-learn.org/stable/index.html) interface for PyTorch models
- [PyRo](https://pyro.ai/): deep universal probabilistic programming with PyTorch
- [Kornia](https://kornia.org/): differentiable computer vision library
- [DGL](http://dgl.ai/): package for deep learning on graphs
- [PyGeometric](https://pytorch-geometric.readthedocs.io/): geometric deep learning extension library for PyTorch
- [PyTorch-BigGraph](https://torchbiggraph.readthedocs.io/en/latest/?badge=latest): a distributed system for learning graph embeddings for large graphs
- [Torchmeta](https://tristandeleu.github.io/pytorch-meta/): datasets and models for few-shot-learning/meta-learning
- [PyTorch3D](https://pytorch3d.org/): library for deep learning with 3D data
- [learn2learn](http://learn2learn.net/): meta-learning model implementations
- [higher](https://higher.readthedocs.io/en/latest/): higher-order (unrolled first-order) optimization
- [Captum](https://captum.ai/): model interpretability and understanding
- [PyTorch summary](https://github.com/amarczew/pytorch_model_summary): Keras style summary for PyTorch models
- [Catalyst](https://catalyst-team.github.io/catalyst/): PyTorch framework for Deep Learning research and development
- [Poutyne](https://poutyne.org/): a simplified framework for PyTorch and handles much of the ea code needed to train neural networks

### Miscellaneous
- [Awesome-Pytorch-list](https://github.com/bharathgs/Awesome-pytorch-list): a comprehensive list of pytorch related content on github,such as different models,implementations,helper libraries,tutorials etc. 
- [DoWhy](https://microsoft.github.io/dowhy/): causal inference combining causal graphical models and potential outcomes
- [CausalML](https://causalml.readthedocs.io/en/latest/?badge=latest): a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent research
- [NetworkX](https://networkx.github.io/documentation/stable/): creation, manipulation, and study of complex networks/graphs
- [Gym](https://gym.openai.com/): toolkit for developing and comparing reinforcement learning algorithms
- [Polygames](https://github.com/facebookincubator/polygames): a platform of zero learning with a library of games
- [Mlxtend](http://rasbt.github.io/mlxtend/): extensions and helper modules for data analysis and machine learning
- [NLTK](https://www.nltk.org/): a leading platform for building Python programs to work with human language data
- [PyCaret](https://pycaret.org/): low-code machine learning library
- [dabl](https://dabl.github.io/dev/): baseline library for data analysis
- [OGB](https://ogb.stanford.edu/): benchmark datasets, data loaders and evaluators for graph machine learning
- [AI Explainability 360](https://aix360.mybluemix.net/): a toolkit for interpretability and explainability of datasets and machine learning models
- [SDV](https://sdv.dev/SDV/): synthetic data generation for tabular, relational, time series data
- [SHAP](https://github.com/slundberg/shap): game theoretic approach to explain the output of any machine learning mode
- [TextBlob](https://textblob.readthedocs.io/en/dev/): a Python (2 and 3) library for processing textual data


## Resources
### Datasets:
- [Google Datasets](https://cloud.google.com/public-datasets): high-demand public datasets
- [Google Dataset Search](https://datasetsearch.research.google.com): a search engine for freely-available online data
- [OpenML](https://www.openml.org/): online platform for sharing data, ML algorithms and experiments
- [DoltHub](https://www.dolthub.com/): data collaboration with Dolt
- [OpenBlender](https://www.openblender.io): live-streamed open data sources
- [Data Portal](https://dataportals.org/): a comprehensive list of open data portals from around the world
- [Activeloop](https://www.activeloop.ai/): unstructured dataset management for TensorFlow/PyTorch

### Libraries:
- [Best-of Machine Learning with Python](https://github.com/ml-tooling/best-of-ml-python): a ranked list of awesome machine learning Python libraries

### Readings:
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design) by Chip Huyen
- [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) by Martin Zinkevich
- [Awesome Data Science](https://github.com/academic/awesome-datascience): an awesome data science repository to learn and apply for real world problems

### Other ML/DL Templates:
- [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science): a logical, reasonably standardized, but flexible project structure
- [PyTorch Template Project](https://github.com/victoresque/pytorch-template): PyTorch deep learning project template


## Authors
* Xiaotian Duan (Email: xduan7 at gmail.com)


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for more details.

