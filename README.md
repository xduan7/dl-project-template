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
    pylint>=2.4

optional:
    poetry>=0.12
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
- [x] minimal setup for hyper-parameter optimization
    - [x] configuration file
- [ ] flowchart for DL/ML projects with packages
- [ ] documentation
    - [ ] getting started
    - [ ] module docstrings


## Extra Packages


## Resources
Readings:
 - [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design) by Chip Huyen
 - [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) by Martin Zinkevich
 
Other Machine Learning Template Repos:
- [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science)
- [PyTorch Template Project](https://github.com/victoresque/pytorch-template)
- [PyTorch Project Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template)


## Authors
* Xiaotian Duan (Email: xduan7 at uchicago.edu)


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for more details.

