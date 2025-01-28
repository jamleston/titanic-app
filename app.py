import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loads
logistic_model = joblib.load('./models/logistic_regression_model.pkl')
decision_tree_model = joblib.load('./models/decision_tree_model.pkl')
random_forest_model = joblib.load('./models/random_forest_model.pkl')
gradient_boosting_model = joblib.load('./models/gradient_boosting_model.pkl')

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio(
    '',
    ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
)
st.sidebar.divider()
st.sidebar.markdown(
    """
    **Note:**
    *Random Forest* has the highest accuracy (0.83) and is recommended for reliable predictions
    """
)

# Models
model_mapping = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "Gradient Boosting": gradient_boosting_model
}

selected_model = model_mapping[model_choice]

# App
st.title("Titanic Survival Prediction 🚢☠️")
st.header("Enter passenger data:")

pclass = st.selectbox("Passenger Class (Pclass):", [1, 2, 3])
sex = st.selectbox("Sex:", ["Male", "Female"])
age = st.slider("Age:", 0, 100, 25)
sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp):", 0, 8, 0)
parch = st.slider("Number of Parents/Children Aboard (Parch):", 0, 6, 0)
fare = st.number_input("Ticket Fare (Fare):", min_value=0.0, max_value=500.0, value=30.0)
embarked = st.selectbox("Port of Embarkation:", ["C", "Q", "S"])

# Converting data
sex_encoded = 1 if sex == "Female" else 0

embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_mapping[embarked]

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Dataframe to show
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_encoded],
    "Age": [age],
    "Fare": [fare],
    "Embarked": [embarked_encoded],
    "FamilySize": [family_size],
    "IsAlone": [is_alone]
})

st.divider()
st.header("Passenger Details:")
st.write("*please check your data*", input_data)
st.divider()

if st.button("Predict Survival"):
    prediction = selected_model.predict(input_data)[0]
    probability = selected_model.predict_proba(input_data)[0][1]

    survival_probability = probability * 100
    death_probability = (1 - probability) * 100

    if prediction == 1:
        st.success(f"The passenger is predicted to survive with a probability of {survival_probability:.2f} 🚢💃")
    else:
        st.error(f"The passenger is predicted to die with a probability of {death_probability:.2f} 🚢☠️")