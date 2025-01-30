# Titanic Survival Prediction App

## Project Overview

This project aims to predict whether a passenger would survive the Titanic disaster based on various features such as age, ticket class, fare, and more. The prediction is implemented using multiple machine learning models and presented as an interactive Streamlit web application.

## Features

**EDA:**
- Data Insights:
    - The Titanic dataset includes passenger details such as `Age`, `Sex`, `Pclass`, `Fare`, `Embarked`, and family-related features (`SibSp`, `Parch`).
    - Missing data for `Age`, `Embarked`, and `Cabin` was handled appropriately during preprocessing.
- Key Findings:
    - Higher-class passengers (`Pclass=1`) and females had a significantly higher survival rate.
    - Higher ticket fares were correlated with better survival odds.
    - Passengers traveling alone had a lower chance of survival compared to those traveling with family.

**Preprocessing:**
- Handling Missing Data
    - `Age`: Missing values were imputed using the median to maintain data distribution.
    - `Embarked`: Missing values were filled with the most common embarkation port (mode).
    - `Cabin`: Since most values were missing, the column was dropped.
- Feature Engineering
    - Encoding Categorical Variables:
        - `Sex`: Converted into binary format (`0 = Male, 1 = Female`).
        - `Embarked`: Mapped to numerical values (`C = 0, Q = 1, S = 2`).
    - New Features Created:
        - `FamilySize` = `SibSp + Parch + 1` (Total number of family members onboard).
        - `IsAlone` = `1 if FamilySize = 1, else 0`.
- Feature Scaling:
    - `Age` and `Fare` were standardized using `StandardScaler` to improve model performance.

**Model Development:**
- Four models were built and tested to predict survival:
    - Logistic Regression
    - Decision Tree
    - Random Forest
        - Achieved the highest accuracy (0.83), making it the most reliable model for predictions.
    - Gradient Boosting

**Streamlit App:**
- An interactive Streamlit-based web application allows users to:
    - Input passenger details such as age, ticket class, and family size.
    - Choose between different machine learning models.
    - View predictions and probabilities of survival.
- Sidebar Features:
    - Model Selection: Choose between Logistic Regression, Decision Tree, Random Forest, or Gradient Boosting.
    - Note: A message informs users that Random Forest has the highest accuracy (0.83) and is recommended for reliable predictions.
- Passenger Input Section:
    - Features sliders, dropdowns, and input fields for Age, Passenger Class (Pclass), Sex, Embarked Port, Number of Siblings/Spouses (SibSp), and Ticket Fare.

## Technologies Used

- **Languages**: Python
- **Libraries**:
    - **Pandas, NumPy** for data manipulation.
    - **Matplotlib, Seaborn** for data visualization.
    - **Scikit-learn** for machine learning models.
    - **Streamlit** For building an interactive web application to visualize data and make predictions
- **Tools**:
    - **Jupyter Notebook** for development and analysis.
    - **Joblib** for model serialization.

## Installation

To run the Titanic Survival Prediction App locally:

1. Clone the repository:
```
git clone https://github.com/jamleston/titanic-app
cd titanic-app
```
2. Run the Streamlit app:
```
streamlit run app.py
```
3. Open the app in your browser at `http://localhost:8501`.

## Usage

- Open the web app and enter passenger details in the input fields.
- Check your data in visualization part (table before button)
- Select a machine learning model from the sidebar.
- Click "Predict Survival" to see the prediction and survival probability.

## Repository Structure

```
├── titanic_preprocessed.csv    # Preprocessed dataset
├── models/                     # Trained machine learning models
│   ├── model_DT.ipynb          # and notebooks
│   ├── model_GB.ipynb
│   ├── model_LR.ipynb
│   ├── model_RF.ipynb
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── gradient_boosting_model.pkl
├── app.py                      # Streamlit application
├── analysis.ipynb              # Notebook for EDA
├── prep.ipynb                  # Notebook for data preprocessing
└── README.md                   # Project documentation
```

## Developed by
- [Valeriia Alieksieienko](https://github.com/jamleston)