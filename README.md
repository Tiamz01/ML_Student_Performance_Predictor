# ML Student Performance Prediction

Welcome to the ML Education Performance Prediction project! This machine learning application seeks to predict students' math scores based on various educational factors and personal attributes. By leveraging the power of machine learning, we aim to uncover the underlying factors that influence student performance and develop predictive models for future assessments.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies]
3. [Key Components](#key-components)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Data Ingestion](#data-ingestion)
    - [Custom Logger And Exception Handler](#identify-script-and-line-of-error-and-log-the-steps)
    - [Data Transformation](#data-transformation)
    - [Model Training](#model-training)
    - [Flask Deployment](#flask-deployment)
4. [Detailed Instructions](#detailed-instructions)
    - [Data Ingestion](#data-ingestion-details)
    - [Data Transformation](#data-transformation-details)
    - [Exploratory Data Analysis (EDA)](#eda-details)
    - [Model Training](#model-training-details)
    - [Flask Deployment](#flask-deployment-details)
7. [Conclusions](#conclusions)

## Project Overview
Understanding and predicting student performance is a critical aspect of educational research. This project provides a comprehensive pipeline for machine learning enthusiasts and educational researchers to work with education performance data.

## Dependencies
* Pandas
* Numpy
* Seaborn 
* Matplotlib
* Scikit-learn
* Catboost
* Xgboost
* Dill
* Flask
* Fickle

### Key Features
- **EDA (Exploratory Data Analysis):** The EDA component uncovers valuable insights from the dataset, highlighting the relationships between various attributes and the student performance indicators.

- **Data Ingestion:** This component allows users to load and split the dataset into training and test sets. It also performs initial data cleansing.

- **Data Transformation:** This component handles data preprocessing, including imputing missing values and encoding categorical features using one hot encoder for use in machine learning models.

- **Model Training:** Here, we explore various machine learning models to find the best predictor for student math scores. It includes model evaluation and hyperparameter tuning.

- **Flask Deployment:** A Flask web application enables users to input student attributes for prediction. The trained model processes this input and returns a predicted math score. See Flask documentation for how to use flask.

## Key Components

### EDA (Exploratory Data Analysis)
- **Description:** The EDA component uncovers valuable insights from the dataset. It analyzes various factors like gender, race/ethnicity, parental level of education, lunch, and test preparation course to understand their impact on student performance. Visualizations and data checks help reveal key patterns and correlations in the data.

- **Python Script:** `STUDENT PERFORMANCE EDA.ipynb`
- **How to Use:** load the `STUDENT PERFORMANCE EDA.ipynb` notebook to explore your dataset's features, conduct data checks, and visualize the relationships between variables. It provides a deep understanding of how various factors affect student performance.

### Data Ingestion
- **Description:** The data ingestion component loads the dataset from a CSV file, splits it into training and test sets, and stores them in specified paths. This prepares the data for further processing.  

- **Python Script:** `data_ingestion.py`
- **How to Use:** Use the `data_ingestion.py` script to load your dataset, split it into training and test data, and save them to specified paths. The script handles the essential data setup. The model and preprocessing steps are then saved to model.pkl and preprocessor.pkl respectively.

### Data Transformation
- **Description:** Data transformation is the preprocessing phase of the project. It includes handling missing values, encoding categorical features, and scaling numerical data. A transformation pipelines is used handle the numerical features and categorical features. the pipelines is then combined into a single pipeline using a column transform. The resulting data is ready for training machine learning models.

- **Python Script:** `data_transformation.py`
- **How to Use:** Customize the preprocessing steps in `data_transformation.py` to match the dataset. Define how missing values are handled and how categorical features are encoded. This script prepares the data for model training.

### Model Training
- **Description:** This component explores several machine learning models to predict student math scores. It evaluates models' performance and selects the best one for predictions. Hyperparameter tuning is also performed to optimize the model. Grid Search cross validation is used to obtained the best model which is Lasso for this project.

- **Python Script:** `model_trainer.py`
- **How to Use:** Utilize the `model_trainer.py` script to train and evaluate machine learning models on your dataset. The script includes a dictionary of models and parameters for customization. This is where you identify the best model for predictions.

### Flask Deployment
- **Description:** The Flask web application component provides an interface for users to input student attributes, which are sent to the trained model for prediction. The application accepts POST requests, processes the data, and returns predicted math scores to users.

- **Python Script:** `app.py`
- **How to Use:** Follow the instructions in the [Flask Deployment Details](#flask-deployment-details) section to set up and customize the Flask web application for your project.

## Getting Started
To get started with the ML Education Performance Prediction project, follow these steps:

2. Install the necessary Python dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Follow the instructions in the [Detailed Instructions](#detailed-instructions) section for each key component to customize and execute the scripts.

4. Customize the project for your specific dataset and use case.

## Detailed Instructions

### Data Ingestion Details
The data ingestion component (`data_ingestion.py`) is responsible for loading your dataset and splitting it into training and test sets. Follow these steps:

1. Customize the script to specify the path to your dataset file.
2. Define how you want to split the data, including the proportion of training and test data.
3. Execute the script to load and split your data. The resulting datasets will be saved to the specified paths.

### Data Transformation Details
The data transformation component (`data_transformation.py`) handles data preprocessing tasks. Customize this script to match your dataset:

1. Define how missing values should be imputed. You can choose to remove rows with missing values or fill them with appropriate values.
2. Specify how categorical features should be encoded (e.g., one-hot encoding or label encoding).
3. Scale numerical features if necessary.

### EDA Details
Exploratory Data Analysis (EDA) is crucial for understanding your dataset and the relationships between variables. The EDA script (`eda.py`) provides a comprehensive analysis:

1. Use the script to visualize data distributions, relationships between variables, and any potential outliers.
2. Conduct data checks, such as detecting missing values, duplicates, and data types.
3. Interpret the visualizations and insights derived from the EDA to understand the dataset's nuances.

### Model Training Details
The model training component (`model_trainer.py`) explores different machine learning models and evaluates their performance. Follow these steps:

1. Customize the script by specifying your dataset and data preprocessing steps.
2. Define a dictionary of machine learning models and their associated hyperparameters.
3. Run the script to train and evaluate the models. The script will identify the best-performing model for predictions.

### Flask Deployment Details
The Flask web application (`app.py`) enables users to input student attributes for prediction via a POST request. Customize the Flask deployment as follows:

1. Ensure you have Flask and the necessary dependencies installed (refer to `requirements.txt`).
2. Modify the model loading section to load your trained model. Replace the sample model with your own.
3. Define the route for capturing POST requests and processing the data.
4. Customize the response format to return predictions..


## Conclusions
- Student's Performance is related to various factors, including lunch, race/ethnicity, parental level of education, and gender.
- Females lead in pass percentage and also achieve top scores.
- Student's performance is not significantly affected by test preparation courses.
- Completing a test preparation course is beneficial, leading to higher scores.
- Exploratory Data Analysis (EDA) revealed insights such as the impact of parental education on students' performance and the relationships between different variables and scores.

