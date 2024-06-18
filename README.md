# Loan_Risk_Prediction

This project involves building a Machine Learning model to predict if a client is high risk or low risk for loan approval. The dataset is provided in JSON format and the model predicts the `Risk_Flag` which indicates whether the client is high risk (1) or low risk (0).

## Project Structure

# Loan Risk Prediction

This project involves building a Machine Learning model to predict if a client is high risk or low risk for loan approval. The dataset is provided in JSON format and the model predicts the `Risk_Flag` which indicates whether the client is high risk (1) or low risk (0).

## Project Structure

```plaintext
loan-risk-prediction/
│
├── data/
│   └── loan_approval_dataset.json
│
├── reports/
│   └── ML_Model_Report.pdf
│
├── scripts/
│   ├── model_training.py
│   └── create_pdf_report.py
│
├── images/
│   ├── feature_importances.png
│   ├── pairplot.png
│   ├── corr_matrix.png
│   └── dist_risk_flag_corrected.png
│
├── README.md
└── requirements.txt
```

## Overview
This project includes:
* Data Exploration: Understanding the dataset and visualizing key aspects.
* Model Training: Building a Machine Learning model to predict loan risk.
* Model Evaluation: Evaluating the model's performance using various metrics.
* Report Generation: Creating a detailed PDF report of the analysis and results.

## Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/radhika3131/loan_risk_prediction.git

cd loan_risk_prediction

```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Dataset
Ensure that the dataset loan_approval_dataset.json is in the data/ directory. If not, place it there.

## Usage

### 1. Data Exploration and Model Training
Run the model_training.py script to explore the data, train the model, and generate visualizations:

```
python scripts/Model_Training.py

```
This script will:
* Load and preprocess the data.
* Train a RandomForestClassifier to predict the Risk_Flag.
* Generate and save visualizations such as feature importances, pair plots, and a correlation matrix.
* 
### 2. Generate PDF Report
After running the model training script, generate the PDF report using the create_pdf_report.py script:

```
python scripts/Pdf_Report.py

```
This script will:

* Compile the generated visualizations and model evaluation metrics into a PDF report.
* Save the report in the reports/ directory.

### Detailed Description of Scripts
#### Model_Training.py
This notebook performs the following steps:

1. Data Loading: Loads the dataset from the data/loan_approval_dataset.json file.
2. Data Preprocessing:
  * Handles missing values.
  * Encodes categorical variables.
  * Scales numerical features.
3. Model Training:
  * Splits the data into training and testing sets.
  * Trains a RandomForestClassifier.
  * Evaluates the model using a confusion matrix and classification report.
4. Visualization: Generates and saves plots for feature importances, pair plots, and the correlation matrix.

#### Pdf_Report.py
This script compiles the visualizations and model performance metrics into a PDF report.


### Results
The final model's performance and insights are documented in the ML_Model_Report.pdf report.

### Conclusion
This project showcases a complete workflow for building and evaluating a machine learning model to predict loan risk. The use of visualizations helps in understanding the data and the model's performance, while the PDF report provides a comprehensive overview of the findings.

By successfully completing this project, I have demonstrated my ability to handle complex datasets, build robust machine learning models, and communicate the results effectively. 


