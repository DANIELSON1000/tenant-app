# -*- coding: utf-8 -*-
"""
Tenant Rent Prediction Web App (Cloud-ready, joblib only)
Author: GODSON
"""

import streamlit as st
import pandas as pd
import os
import joblib

# -------------------------------
def load_model():
    """Load the ML model (joblib only)."""
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "tenant_model.joblib")
    if not os.path.exists(MODEL_FILE):
        st.error("❌ Model file not found. Please upload tenant_model.joblib.")
        st.stop()
    return joblib.load(MODEL_FILE)

def load_history():
    """Load existing history CSV or create empty DataFrame."""
    HISTORY_FILE = os.path.join(os.path.dirname(__file__), "tenant_history.csv")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
    else:
        history_df = pd.DataFrame(columns=[
            'BHK','Size','Bathroom','Furnishing Status','Tenant Preferred',
            'City','Point of Contact','Area Locality','Posted On','Area Type','Floor','Predicted Rent',
            'Tenant Name', 'Telephone Number', 'Previous Payment Date'  # New columns
        ])
    return history_df, HISTORY_FILE

# -------------------------------
def main():
    # Page config
    st.set_page_config(page_title="Tenant Rent Prediction", page_icon="🏠", layout="wide")

    # Load model and history
    model = load_model()
    history_df, HISTORY_FILE = load_history()

    # App title
    st.markdown(
        """
        <h1 style="text-align:center; color:#2E86C1;">🏡 Tenant Rent Prediction System</h1>
        <p style="text-align:center; color:#7D3C98; font-size:18px;">
        Enter tenant & property details to predict monthly rent and manage records.
        </p>
        """, unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar inputs
    st.sidebar.header("🔧 Tenant & Property Details")
    BHK = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2)
    Size = st.sidebar.number_input("Size (sq.ft)", min_value=100, max_value=10000, value=950)
    Bathroom = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    Furnishing_Status = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
    Tenant_Preferred = st.sidebar.selectbox("Tenant Preferred", ["Bachelors", "Family", "Bachelors/Family"])
    City = st.sidebar.text_input("City")
    Point_of_Contact = st.sidebar.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])
    Area_Locality = st.sidebar.text_input("Area Locality")
    Posted_On = st.sidebar.date_input("Posted On")
    Area_Type = st.sidebar.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
    Floor = st.sidebar.text_input("Floor (e.g. '5 out of 10')", "5 out of 10")
    
    # New input fields for tenant information (not used in ML model)
    st.sidebar.header("👤 Tenant Information")
    Tenant_Name = st.sidebar.text_input("Tenant Name")
    Telephone_Number = st.sidebar.text_input("Telephone Number")
    Previous_Payment_Date = st.sidebar.date_input("Previous Payment Date")

    # Prepare input for ML model (without the new fields)
    input_data = pd.DataFrame({
        'BHK':[BHK],
        'Size':[Size],
        'Bathroom':[Bathroom],
        'Furnishing Status':[Furnishing_Status],
        'Tenant Preferred':[Tenant_Preferred],
        'City':[City],
        'Point of Contact':[Point_of_Contact],
        'Area Locality':[Area_Locality],
        'Posted On':[str(Posted_On)],
        'Area Type':[Area_Type],
        'Floor':[Floor]
    })

    # Predict button
    if st.sidebar.button("🔮 Predict Rent"):
        predicted_rent = model.predict(input_data)[0]

        # Save to history with additional tenant information
        history_entry = input_data.copy()
        history_entry['Predicted Rent'] = predicted_rent
        history_entry['Tenant Name'] = Tenant_Name
        history_entry['Telephone Number'] = Telephone_Number
        history_entry['Previous Payment Date'] = str(Previous_Payment_Date)
        
        history_df = pd.concat([history_df, history_entry], ignore_index=True)
        history_df.to_csv(HISTORY_FILE, index=False)

        # Show prediction
        st.markdown(
            f"""
            <div style="background-color:#D6EAF8; padding:20px; border-radius:12px;">
                <h2 style="color:#154360;"> Predicted Monthly Rent:  {predicted_rent:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True
        )

        if predicted_rent > 90000:
            st.error(" High rent: tenant may struggle to pay on time.")
        else:
            st.success("Rent is reasonable: tenant likely to pay on time.")

    # Old predictions table with delete
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

# -------------------------------
if __name__ == "__main__":
    main()