# AgroTech Innovations Machine Learning pipeline

Name: Liu Yujia
Email: liuyujia2002@gmail.com

## Overview of folder
```
|-- .github/
|   |-- github-actions.yml
|-- src/
|   |-- data/
|   |-- load_data.py
|   |-- main.py
|   |-- model.py
|   |-- preprocess.py
|
|-- config.json
|-- eda.ipynb
|-- README.md
|-- requirements.txt
|-- run.sh
|-- plant_stage_model_tree.pkl
|-- plant_stage_modelNN.pkl
|-- temperature_model_tree.pkl
|-- temperature_modelNN.pkl
```
In the `src` folder, the `data` folder is to store the downloaded data. The 4 python files is the machine learning pipeline. The `config.json` allows for easy configuration of the pipeline to enable easy experimentation of different algorithms and parameters as well as ways of processing data. `run.sh` is a bash script that triggers the pipeline run. The files with `.pkl` extensions are the trained model.
## Setup Instructions

Follow these steps to set up the project on your local machine:

### Prerequisites

- Python 3.x (>= 3.6 recommended)
- pip (Python package installer)

### Step 1: Clone the repository

```bash

git clone https://github.com/LiuYJ2002/aiip5-liu-yujiai-640I.git
```
Open a virtual environment(optional):
```bash
python -m venv venv
.\venv\Scripts\activate.bat
```
### Step 2: Install dependencies

```bash
cd aiip5-liu-yujia-640I
pip install -r requirements.txt
```

### Step 3: Modify parameters
In the `config.json` file, you can modify parameters for model training. As the task involves using 2 different model types, I will use their `config.json` files for explanation. I chose `Random forest` and `Neural networks` model for the task. 

This is the config file for the `Neural network`:
```
{
    "data": {
      "missing": true,
      "duplicates": true,
      "outliers": true,
      "normalise": true
    },
    "temperature_model": {
      "type": "MLPRegressor",
      "params": {
        "hidden_layer_sizes": [128, 64],
        "activation": "relu",
        "solver": "adam",
        "max_iter": 400,
        "learning_rate": "adaptive",
        "alpha": 0.001,
        "batch_size": "auto",
        "early_stopping": true,
        "n_iter_no_change": 30
      },
      "tuning": {
        "type": "grid",
        "cv": 3,
        "search_space": {
          "hidden_layer_sizes": [[128, 64], [256, 128, 64], [512, 256, 128, 64]],
          "activation": ["relu", "tanh"],
          "solver": ["adam"],
          "learning_rate": ["adaptive"],
          "alpha": [0.001, 0.01, 0.1],
          "batch_size": ["auto", 32, 64, 128]
        }
      }
    },
    "plant_stage_model": {
      "type": "MLPClassifier",
      "params": {
        "hidden_layer_sizes": [128, 64],
        "activation": "relu",
        "solver": "adam",
        "max_iter": 400,
        "learning_rate": "adaptive",
        "alpha": 0.001,
        "batch_size": "auto",
        "early_stopping": true,
        "n_iter_no_change": 30
      },
      "tuning": {
        "type": "grid",
        "cv": 3,
        "search_space": {
          "hidden_layer_sizes": [[128, 64], [256, 128, 64], [512, 256, 128, 64]],
          "activation": ["relu", "tanh", "logistic"],
          "solver": ["adam"],
          "learning_rate": ["adaptive"],
          "alpha": [0.001, 0.01, 0.1],
          "batch_size": ["auto", 32, 64, 128]
        }
      }
    }
}
```
In the `data` section, setting a key to be `true` will the enable the corresponding data processing step to be executed. For example, setting `outliers` to be `true` will result in removing outliers during the data processing step.

Under `temperature model` you can set the various parameters to train the regressor model for temperature prediction.

Under `plant_stage_model` you can set the various parameters to train the classification model for Plant type-stage prediction.

Under `tuning`, you can set the `search space` for hyperparameter tunings using different methods `random, grid, bayesian` under `type`.

Below is the second config file I used for my second random forest model that I explored:

```
{
    "data": {
      "missing": true,
      "duplicates": true,
      "outliers": true,
      "normalise": true
    },
    "temperature_model": {
      "type": "RandomForestRegressor",
            "params": {
                "n_estimators": 100,
                "max_depth": 10
            },
            "tuning": {
                "type": "grid",
                "cv": 3,
                "search_space": {
                    "n_estimators": [100, 250, 500],
                    "max_depth": [10, 20, 30, null],
                    "min_samples_split": [5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", null],
                    "bootstrap": [true, false]
                }
            }
    },
    "plant_stage_model": {
      "type": "RandomForestClassifier",
            "params": {
                "n_estimators": 100,
                "max_depth": 10
            },
            "tuning": {
                "type": "grid",
                "cv": 3,
                "search_space": {
                    "n_estimators": [100, 250, 500],
                    "max_depth": [10, 20, 30, null],
                    "min_samples_split": [5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", null],
                    "bootstrap": [true, false],
                    "criterion": ["gini", "entropy"]
                }

            }
  
    }
}
```
The usage is similar and noticed I just needed to change the `type` to `RandomForestRegressor` and `RandomForestClassifier`. You can easily change the hyparameters and search space using similar logic. 

### Step 4 : pipeline execution

To train your model, first download your data into `src/data`. In this problem I downloaded the database from the following URL:
https://techassessment.blob.core.windows.net/aiip5-assessment-data/agri.db

Next to run the training pipeline execute:

```
bash run.sh
```

## Logical steps/flow of pipeline

1. Data Ingestion: The fetch data function from `load.py` is triggered to load the data into a pandas dataframe.

2. Preprocessing & Feature Engineering: Using the values set in the `config.json` file, data is preprocessed and categorical values are one hot encoded in the `preprocess.py` file. They are later split for training and testing in a 80:20 ratio.

3. Model Training: For this task, 2 different models are trained for each temperature prediction and plant type-stage classification according to the `config.json` file in the `model.py` file.

4. Model Evaluation: As `hyperparameter` tuning is deployed, the model with the best results will have its hyperparameters saved and printed to the terminal. For temperature, `MSE` is used as the evaluation metrics and for plant type-stage classification: `accuracy, precision, recall, F1-score` is used.

5. Storage: Models are saved as `.pkl` files

## Overview of EDA

In the EDA I discovered that there are a few problems with the dataset: `duplicate rows, missing values, outliers, incorrect data types for values in Nutrient n, p, k columns, inconsistancy in capitalisations for categorical values`. Thus during feature engineering, I targetted this issues to resolve them and also did `one hot encoding` of categorical variables and `normalisation` of numeric values. Only some of the features follow normal distribution and there is no strong correlation nor relationships between the feature values. Refer more to the `.ipynb` file for more details and statistics/graphs too.

## Features
Duplicate rows are also removed
Features  | Processing done
------------- | -------------
System Location Code  | one hot encoding
Previous Cycle Plant Type  | one hot encoding, standardise values to lower case
Plant Type  | one hot encoding, standardise values to lower case
Plant Stage  | one hot encoding, standardise values to lower case
Temperature Sensor (°C)  | handle missing values and outliers, normalise
Humidity Sensor (%)  | handle missing values and outliers, normalise
Light Intensity Sensor (lux)  | handle missing values and outliers, normalise
CO2 Sensor (ppm)  | normalise
EC Sensor (dS/m)  | handle outliers, normalise
O2 Sensor (ppm)  | handle outliers, normalise
Nutrient N Sensor (ppm)  | handle missing values and outliers, normalise, correct its data type
Nutrient P Sensor (ppm)  | handle missing values, normalise, correct its data type
Nutrient K Sensor (ppm)  | handle missing values, normalise, correct its data type
pH Sensor  | handle outliers, normalise
Water Level Sensor (mm)  | handle missing values and outliers, normalise
plant type-stage  | merged for the classification task
## Choice of model

First task is temperature prediction. A regression model is needed so I picked `RandomForestRegressor` and `Neural network (MLPRegressor)`.

First task is plant type-stage prediction. A classification model is needed so I picked `RandomForestClassifier` and `Neural network(MLPClassifier)`

Random Forest is an ensemble method that can handle non-linear relationships. It works by constructing multiple decision trees and averaging their predictions, making it robust. Neural Networks are selected for their ability to capture complex non-linear relationships by using multiple layers of neurons to learn relationships. Both can handle regression and classification tasks thus they are selected

## Model evaluation

### Task 1: Temperature prediction

#### Random forest regressor
After hyperparameter tuning, this is the Best parameters found for RandomForestRegressor: 
```
{'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': False}
```
```
Temperature Prediction MSE: 0.4800210789070718
```

#### MLPRegressor
After hyperparameter tuning, this is the Best parameters found for MLPRegressor: 
```
Best parameters found for MLPRegressor: {'activation': 'relu', 'alpha': 0.1, 'batch_size': 64, 'hidden_layer_sizes': [512, 256, 128, 64], 'learning_rate': 'adaptive', 'solver': 'adam'}
```
```
Temperature Prediction MSE: 1.0114102612332727
```

Mean squared error is used to evaluate the model. As random forest regressor has a lower MSE, it performs better with a lower MSE. 

### Task 2: Plant type-stage prediction

#### Random forest classifier
After hyperparameter tuning, this is the Best parameters found for RandomForestClassifier: 
```Best parameters found for RandomForestClassifier: {'bootstrap': False, 'criterion': 'gini', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}
```
```
Plant Type-Stage Classification Accuracy: 0.7598488936859147
Precision: 0.7625889315095852, Recall: 0.7598488936859147, F1-Score: 0.7593072395306191
```

#### MLPClassifier
After hyperparameter tuning, this is the Best parameters found for MLPClassifier: 

```
Best parameters found for MLPClassifier: {'activation': 'logistic', 'alpha': 0.01, 'batch_size': 'auto', 'hidden_layer_sizes': [128, 64], 'learning_rate': 'adaptive', 'solver': 'adam'}
```
```
Plant Type-Stage Classification Accuracy: 0.7737 Precision: 0.7777675738223253, Recall: 0.7737, F1-Score: 0.7720502802522011
```

The MLPClassifier outperforms the RandomForestClassifier in terms of accuracy. It also has better F1 score suggesting better precision and recall. Thus MLPClassifier has a slightly better performance.

## Other considerations for deploying the developed model
1. Model performance monitoring.

2. Periodic data updates and retraining when skew and drift is noticed.

3. Resource efficiency in model inference for real-time predictions.

4. Could leverage cloud services for training and deployment due to the scalability and flexibility offered. (Eg. vertexAI on google cloud)

5. Performance of model could be further improved for this task by expanding the hyperparameter search space and also testing out different potential feature transformation like log, adding exponential terms or merging features.
