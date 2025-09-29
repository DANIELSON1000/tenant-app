# -*- coding: utf-8 -*-
"""
Tenant Rent Prediction Web App with Data Visualization
"""

import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load the saved model
model_path = r"C:\Users\GODSON\Desktop\work\tenant_model.joblib"
loaded_model = joblib.load(open(model_path, 'rb'))

# Streamlit page config
st.set_page_config(page_title="Tenant Rent Prediction", page_icon="🏠", layout="wide")

# Title
st.markdown(
    """
    <h1 style="text-align:center; color:#2E86C1;">🏡 Tenant Rent Prediction System</h1>
    <p style="text-align:center; color:#7D3C98; font-size:18px;">
    Enter tenant & property details to predict monthly rent and manage records.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# Initialize session state for old predictions
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'BHK','Size','Bathroom','Furnishing Status','Tenant Preferred',
        'City','Point of Contact','Area Locality','Posted On','Area Type','Floor','Predicted Rent'
    ])

# Sidebar input
st.sidebar.header("🔧 Input Features")
BHK = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2)
Size = st.sidebar.number_input("Size (sq.ft)", min_value=100, max_value=10000, value=950)
Bathroom = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
Furnishing_Status = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
Tenant_Preferred = st.sidebar.selectbox("Tenant Preferred", ["Bachelors", "Family", "Bachelors/Family"])
City = st.sidebar.text_input("City")
Point_of_Contact = st.sidebar.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])
Area_Locality = st.sidebar.text_input("Area Locality")
Posted_On = st.sidebar.date_input("Posted On")
Area_Type = st.sidebar.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
Floor = st.sidebar.text_input("Floor (e.g. '5 out of 10')", "5 out of 10")

# Prepare input
input_data = pd.DataFrame({
    'BHK': [BHK],
    'Size': [Size],
    'Bathroom': [Bathroom],
    'Furnishing Status': [Furnishing_Status],
    'Tenant Preferred': [Tenant_Preferred],
    'City': [City],
    'Point of Contact': [Point_of_Contact],
    'Area Locality': [Area_Locality],
    'Posted On': [str(Posted_On)],
    'Area Type': [Area_Type],
    'Floor': [Floor]
})

# Predict button
if st.sidebar.button("🔮 Predict Rent"):
    prediction = loaded_model.predict(input_data)
    predicted_rent = prediction[0]

    # Add to history
    input_data['Predicted Rent'] = predicted_rent
    st.session_state.history = pd.concat([st.session_state.history, input_data], ignore_index=True)

    # Display prediction
    st.markdown(
        f"""
        <div style="background-color:#D6EAF8; padding:20px; border-radius:12px;">
            <h2 style="color:#154360;">💰 Predicted Monthly Rent: ₹ {predicted_rent:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk scenario
    if predicted_rent > 90000:
        st.error("⚠️ Tenant may struggle to pay on time (High Rent Risk).")
    else:
        st.success("✅ Tenant is likely to pay rent on time (Safe Zone).")

st.markdown("---")

# Section: Rent Histogram
st.subheader("📊 Rent Distribution")
if not st.session_state.history.empty:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(st.session_state.history['Predicted Rent'], bins=10, color="#5DADE2", edgecolor="black")
    ax.set_xlabel("Predicted Rent")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Predicted Rent")
    st.pyplot(fig)
else:
    st.info("No predictions yet to show histogram.")

st.markdown("---")

# Section: Old Predictions Table with Delete
st.subheader("📝 Old Predictions")
if not st.session_state.history.empty:
    st.dataframe(st.session_state.history)

    st.markdown("**Delete a row by index:**")
    delete_index = st.number_input("Row index to delete", min_value=0, max_value=len(st.session_state.history)-1, step=1)
    if st.button("🗑️ Delete Row"):
        st.session_state.history.drop(index=delete_index, inplace=True)
        st.session_state.history.reset_index(drop=True, inplace=True)
        st.success(f"Row {delete_index} deleted successfully.")
else:
    st.info("No old predictions yet.")
