# Student Math Score Prediction Project

## Overview

An end-to-end machine learning pipeline to predict students’ **math scores** using demographic and academic features. The system preprocesses data, trains multiple regression models with hyperparameter tuning, and selects the best model based on **R² score**.

## Features

* **Data Ingestion**: Loads `stud.csv`, performs 80/20 train-test split, saves artifacts.
* **Data Transformation**: Handles missing values, categorical encoding (OneHotEncoder), and scaling (StandardScaler).
* **Model Training**: Trains and tunes multiple regressors using GridSearchCV.
* **Model Selection**: Chooses the best model based on test R² score.
* **Robustness**: Logging and custom exception handling.

## Tech Stack

* Python 3.x
* pandas, numpy, scikit-learn
* XGBoost, CatBoost
* dill (serialization)

## Project Structure

```
src/
├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
├── utils.py
├── exception.py
├── logger.py
artifacts/
stud.csv
README.md
```

## Installation

```bash
git clone <repository-url>
cd <project-directory>
pip install -r requirements.txt
```

## Usage

```bash
python -m src.components.data_ingestion
```

Outputs:

* Trained model: `artifacts/model.pkl`
* Preprocessor: `artifacts/preprocessor.pkl`
* Best model with R² score printed

## Dataset

* **Target**: `math_score`
* **Features**: gender, race_ethnicity, parental education, lunch, test prep, reading score, writing score
* **Size**: 1000 rows

## Models Used

* Linear Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* KNN
* AdaBoost
* XGBoost
* CatBoost



