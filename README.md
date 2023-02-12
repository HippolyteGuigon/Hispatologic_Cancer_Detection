# Hispathologic cancer detection competition in Kaggle 

The goal of this repository is to participate to the Hispathologic cancer detection competition in Kaggle (https://www.kaggle.com/competitions/histopathologic-cancer-detection). The goal of this competition is to develop a Machine Learning algorithm that can detect metastatic cancer in small image patches taken from larger digital pathology scans. 

## Build Status

For the moment, the main pipeline is in place and the user can classify his images, and the user can choose between two models that are Convolutional Neural Network and Transformer

The next objectives of this repositories is only to implement more models (Clustering techniques, other Deep Learning methods) but also developping more options for the user in terms of metrics and visualization possibilities.

Throughout its construction, if you see any improvements that could be made in the code, do not hesitate to reach out at 
Hippolyte.guigon@hec.edu

## Code style 

The all project was coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

* This project uses a specific conda environment, to get it, run the following command: ```conda env create -f hispathologic_cancer_environment.yml``` 

* To install all necessary libraries, run the following code: ```pip install -r requirements.txt```

* This project has its own package that is used. To get it, run the following command: ```python install setup.py```

## Screenshot 

![alt text](https://github.com/HippolyteGuigon/Hispatologic_Cancer_Detection/blob/master/ressources/metastatic-breast-cancer.jpeg)

Image of metastatic breast cancer cell.

## How to use ? 

There are two ways to run this program:

-The first one is via the Flask application 

* First, run the following command: ```export FLASK_APP=Hispatologic_cancer_detection/app/app.py```

* Then, run ```flask run```

-The second one is to run it via the command line via the following steps

* To run the model, first change the parameters and the configurations you wish to apply in the following path: 
```configs/```

* Then, run the following command: ```python main.py user_name model_train predict``` with user_name being your own name that will then be printed in the logs