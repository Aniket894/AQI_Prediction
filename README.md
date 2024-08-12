# Project Documentation  : -  AQI Prediction 

## ( ML Regression Porject )

# Table of Contents

## Introduction

## Dataset Description

## Project Objectives

## Project Structure

## Data Ingestion

## Data Transformation

## Model Training

## Training Pipeline

## Prediction Pipeline

## Flask

## Logging

## Exception Handling

## Utils

## Conclusion


# 1. Introduction

The project aims to analyze air quality data based on various features such as date, location, and pollutant levels, using the "Air Quality Dataset." This document provides a comprehensive overview of the project, including its structure, processes, and supporting scripts.


# 2. Dataset Description

**Dataset Name**: Air Quality Dataset

**Description**: The dataset contains information on air quality measurements across different locations. The features include:

**Date**: The date of the measurement.

**Site**: The site where the measurement was taken.

**State**: The state where the measurement was taken.

**Latitude**: The latitude of the measurement site.

**Longitude**: The longitude of the measurement site.

**PM2.5**: The concentration of PM2.5 particles (in µg/m³).

**PM10**: The concentration of PM10 particles (in µg/m³).

**NH3**: The concentration of ammonia (in µg/m³).

**SO2**: The concentration of sulfur dioxide (in µg/m³).

**AQI**: The Air Quality Index value.



# 3. Project Objectives


**Data Ingestion**: Load and explore the dataset.

**Data Transformation**: Clean, preprocess, and transform the dataset for analysis.

**Model Training**: Train various models to analyze and predict air quality metrics.

**Pipeline Creation**: Develop a pipeline for data ingestion, transformation, and analysis.

**Supporting Scripts**: Provide scripts for setup, exception handling, utilities, and logging.


# 4. Project Structure

```AirlineTicketPricePrediction/
│
├── artifacts/
│   ├── (best)model.pkl
│   ├── linearRegression.pkl
│   ├── Lasso.pkl
│   ├── Ridge.pkl
│   ├── ElasticNet.pkl
│   ├── DecisionTreeRegressor.pkl
│   ├── RandomForestRegressor.pkl
│   ├── GradientBoostingRegressor.pkl
│   ├── AdaBoostRegressor.pkl
│   ├── XGBoostRegressor.pkl
│   ├── raw.csv
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── data/
│   │     └── data_date.csv
│   └── AQI Prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_training.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│   ├── plane (1).jpeg
│   └── style.css
│
├── app.py
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py
```


# 5. Data Ingestion

The data ingestion file is used to load the data from the source, split it into training, testing, and raw CSV files, and save them into the artifacts folder.


# 6. Data Transformation

The data transformation file is used to perform exploratory data analysis (EDA), including encoding and preprocessing the data, and saving the processed data.


# 7. Model Training

The model training file is used to train models with various algorithms, save the best model as a pickle file (.pkl) in the artifacts folder, and store all trained models.


![download (6)](https://github.com/user-attachments/assets/d6cc79d5-7fc6-4ab5-a3e1-2b38c225a22e)


# 8. Training Pipeline

This file is used to run the data ingestion, data transformation, and model training scripts in sequence.


# 9. Prediction Pipeline

This file is used to predict outcomes using the best model saved as best_model.pkl and preprocess the data using preprocessor.pkl.


# 10. Static

static/style.css: Provides the theme for the index.html and results.html pages.


# 11. Templates

templates/index.html: Creates a form to get data input from the user.

templates/results.html: Displays the predicted results of the model.


# 12. Flask (app.py)

This file handles form submissions from index.html, predicts results using prediction_pipeline.py, and displays the results.


![Screenshot 08-12-2024 09 57 40](https://github.com/user-attachments/assets/0313f641-5c2e-4ff4-b32d-7c379ed0df82)


![Screenshot 08-12-2024 09 57 52](https://github.com/user-attachments/assets/00bc12ab-1b1f-462f-b3aa-cf759467484a)


# 13. Logger

This file saves logs, recording the execution flow and errors.


# 14. Exception Handling

This file contains the exception handling code for errors, ensuring they are caught and logged appropriately.


# 15. Utils

This file contains utility functions used throughout the project for common tasks, such as creating directories.


# 16. Conclusion
This documentation provides a comprehensive overview of the "Air Quality Analysis" project, covering data ingestion, transformation, model training, pipeline creation, and supporting scripts. The provided code examples illustrate the implementation of various components, ensuring a robust and scalable project structure.



