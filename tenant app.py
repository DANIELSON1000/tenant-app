# -*- coding: utf-8 -*-
"""
Tenant Rent Prediction Web App (Cloud-ready)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# Optional: Use pickle instead of joblib if joblib gives issues
import pickle

# Streamlit page config
st.set_page_config(page_title="Tenant Rent Prediction", page_icon="🏠", layout="wide")

# -------------------------------
# Paths for model and history
MODEL_FILE = os.path.join(os.path.dirname(__file__), "tenant_model.joblib")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "tenant_history.csv")

# -------------------------------
# Load ML model
try:
    import joblib
    model = joblib.load(MODEL_FILE)
except ModuleNotFoundError:
    st.warning("Joblib not found. Falling back to pickle.")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

# -------------------------------
# Load old predictions
if os.path.exists(HISTORY_FILE):
    history_df = pd.read_csv(HISTORY_FILE)
else:
    history_df = pd.DataFrame(columns=[
        'BHK','Size','Bathroom','Furnishing Status','Tenant Preferred',
        'City','Point of Contact','Area Locality','Posted On','Area Type','Floor','Predicted Rent'
    ])

# -------------------------------
# App title
st.markdown(
    """
    <h1 style="text-align:center; color:#2E86C1;">🏡 Tenant Rent Prediction System</h1>
    <p style="text-align:center; color:#7D3C98; font-size:18px;">
    Predict rent, visualize data, and manage old predictions.
    </p>
    """, unsafe_allow_html=True
)
st.markdown("---")

# -------------------------------
# Sidebar: Input form
st.sidebar.header("🔧 Tenant & Property Details")

BHK = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2)
Size = st.sidebar.number_input("Size (sq.ft)", min_value=100, max_value=10000, value=950)
Bathroom = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2)
Furnishing_Status = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
Tenant_Preferred = st.sidebar.selectbox("Tenant Preferred", ["Bachelors", "Family", "Bachelors/Family"])
City = st.sidebar.text_input("City", "Mumbai")
Point_of_Contact = st.sidebar.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])
Area_Locality = st.sidebar.text_input("Area Locality", "Andheri West")
Posted_On = st.sidebar.date_input("Posted On")
Area_Type = st.sidebar.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
Floor = st.sidebar.text_input("Floor (e.g. '5 out of 10')", "5 out of 10")

# -------------------------------
# Prepare input DataFrame
input_data = pd.DataFrame({
    'BHK':[BHK], 'Size':[Size], 'Bathroom':[Bathroom],
    'Furnishing Status':[Furnishing_Status], 'Tenant Preferred':[Tenant_Preferred],
    'City':[City], 'Point of Contact':[Point_of_Contact], 'Area Locality':[Area_Locality],
    'Posted On':[str(Posted_On)], 'Area Type':[Area_Type], 'Floor':[Floor]
})

# -------------------------------
# Predict button
if st.sidebar.button("🔮 Predict Rent"):
    predicted_rent = model.predict(input_data)[0]

    # Add to history and save
    input_data['Predicted Rent'] = predicted_rent
    history_df = pd.concat([history_df, input_data], ignore_index=True)
    history_df.to_csv(HISTORY_FILE, index=False)

    # Show prediction
    st.markdown(
        f"""
        <div style="background-color:#D6EAF8; padding:20px; border-radius:12px;">
            <h2 style="color:#154360;">💰 Predicted Rent: ₹ {predicted_rent:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True
    )

    if predicted_rent > 30000:
        st.error("⚠️ High rent: tenant may struggle to pay on time.")
    else:
        st.success("✅ Rent is reasonable: tenant likely to pay on time.")

# -------------------------------
# Rent Histogram
st.subheader("📊 Rent Distribution")
if not history_df.empty:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(history_df['Predicted Rent'], bins=10, color="#5DADE2", edgecolor="black")
    ax.set_xlabel("Predicted Rent")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Predicted Rent")
    st.pyplot(fig)
else:
    st.info("No predictions yet.")

# -------------------------------
# Old predictions table + delete
st.subheader("📝 Old Predictions")
if not history_df.empty:
    st.dataframe(history_df)

    st.markdown("**Delete a row by index:**")
    delete_index = st.number_input("Row index to delete", min_value=0, max_value=len(history_df)-1, step=1)
    if st.button("🗑️ Delete Row"):
        history_df.drop(index=delete_index, inplace=True)
        history_df.reset_index(drop=True, inplace=True)
        history_df.to_csv(HISTORY_FILE, index=False)
        st.success(f"Row {delete_index} deleted successfully.")
else:
    st.info("No old predictions yet.")
