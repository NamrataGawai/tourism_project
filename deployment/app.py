import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Namrata-gawai/MLOps-Tourism-Package-Prediction", filename="best_tourism_package_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for staff that predicts whether customer will buy the product or not based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to buy product.")

# Collect user input
Age = st.number_input("Age (customer's age)", min_value=18, max_value=150, value=30)
TypeofContact = st.selectbox("Type of Contact (The method by which the customer was contacted (Company Invited or Self Inquiry))", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier (The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3))", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation (Customer's occupation (e.g., Salaried, Freelancer))", ["Salaried", "Freelancer"])
Gender = st.selectbox("Gender (Gender of the customer)", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status (Marital status of the customer (Single, Married, Divorced))", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation (Customer's designation in their current organization)", ["Executive","Manager","Senior Manager", "AVP", "VP"])
ProductPitched = st.selectbox("Product Pitched (The type of product pitched to the customer))",["Basic","Deluxe","King","Standard","Super Deluxe"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (Number of persons visiting the package)", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Followups (Total number of follow-ups by the salesperson after the sales pitch)", min_value=0, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star (Preferred hotel rating by the customer)", min_value=1, max_value=5, value=4)
NumberOfTrips = st.number_input("Number of Trips (Average number of trips the customer takes annually)", min_value=1, value=2)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (Number of children below age 5 accompanying the customer)", min_value=0, value=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (Score indicating the customer's satisfaction with the sales pitch)", min_value=1, max_value=5, value=4)
OwnCar = st.selectbox("Own Car (Whether the customer owns a car (0: No, 1: Yes))", ["No", "Yes"])
Passport = st.selectbox("Whether the customer holds a valid passport (0: No, 1: Yes)", ["No", "Yes"])
MonthlyIncome = st.number_input("Monthly Income (Gross monthly income of the customer)", min_value=0, value=5000)
DurationOfPitch = st.number_input("Duration of Pitch (Duration of the sales pitch delivered to the customer)", min_value=1, value=2)



# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': city_tier
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 0 if OwnCar == "No" else 1
    'Passport': 0 if Passport =="No" else 1,
    'MonthlyIncome': MonthlyIncome,
    'DurationOfPitch': DurationOfPitch



}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "buy product" if prediction == 1 else "not buy product"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
