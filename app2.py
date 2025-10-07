# -*- coding: utf-8 -*-
"""
Tenant Rent Prediction Web App (Cloud-ready, joblib only)
Author: GODSON
Enhanced with due date tracking and better tenant management
"""

import streamlit as st
import pandas as pd
import os
import joblib
from datetime import datetime, timedelta

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
        # Ensure all required columns exist
        required_cols = [
            'BHK','Size','Bathroom','Furnishing Status','Tenant Preferred',
            'City','Point of Contact','Area Locality','Posted On','Area Type','Floor','Predicted Rent',
            'Tenant Name', 'Telephone Number', 'Previous Payment Date', 'Next Payment Due Date', 'Payment Status'
        ]
        for col in required_cols:
            if col not in history_df.columns:
                history_df[col] = ""
    else:
        history_df = pd.DataFrame(columns=[
            'BHK','Size','Bathroom','Furnishing Status','Tenant Preferred',
            'City','Point of Contact','Area Locality','Posted On','Area Type','Floor','Predicted Rent',
            'Tenant Name', 'Telephone Number', 'Previous Payment Date', 'Next Payment Due Date', 'Payment Status'
        ])
    return history_df, HISTORY_FILE

def calculate_due_date(previous_date, frequency_days=30):
    """Calculate next payment due date based on previous payment date."""
    if pd.isna(previous_date) or previous_date == "":
        return ""
    try:
        prev_date = pd.to_datetime(previous_date)
        due_date = prev_date + timedelta(days=frequency_days)
        return due_date.strftime("%Y-%m-%d")
    except:
        return ""

def get_payment_status(due_date):
    """Determine payment status based on due date."""
    if pd.isna(due_date) or due_date == "":
        return "Unknown"
    
    try:
        due = pd.to_datetime(due_date)
        today = pd.to_datetime(datetime.now().date())
        
        if due < today:
            return "Overdue"
        elif due == today:
            return "Due Today"
        elif (due - today).days <= 7:
            return "Due Soon"
        else:
            return "Upcoming"
    except:
        return "Unknown"

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
        <h1 style="text-align:center; color:#2E86C1;">🏡 Tenant Rent Prediction & Management System</h1>
        <p style="text-align:center; color:#7D3C98; font-size:18px;">
        Enter tenant & property details to predict monthly rent and manage tenant records.
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
    
    # New input fields for tenant information
    st.sidebar.header("👤 Tenant Information")
    Tenant_Name = st.sidebar.text_input("Tenant Name")
    Telephone_Number = st.sidebar.text_input("Telephone Number")
    Previous_Payment_Date = st.sidebar.date_input("Previous Payment Date")
    Payment_Frequency = st.sidebar.selectbox("Payment Frequency", 
                                           ["Monthly (30 days)", "Quarterly (90 days)", "Custom"])
    
    if Payment_Frequency == "Custom":
        custom_days = st.sidebar.number_input("Payment frequency (days)", min_value=1, max_value=365, value=30)
        frequency_days = custom_days
    elif "Monthly" in Payment_Frequency:
        frequency_days = 30
    else:  # Quarterly
        frequency_days = 90

    # Calculate due date
    Next_Payment_Due_Date = ""
    if Previous_Payment_Date:
        due_date = Previous_Payment_Date + timedelta(days=frequency_days)
        Next_Payment_Due_Date = due_date.strftime("%Y-%m-%d")

    # Prepare input for ML model
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

        # Calculate payment status
        payment_status = get_payment_status(Next_Payment_Due_Date)

        # Save to history with additional tenant information
        history_entry = input_data.copy()
        history_entry['Predicted Rent'] = predicted_rent
        history_entry['Tenant Name'] = Tenant_Name
        history_entry['Telephone Number'] = Telephone_Number
        history_entry['Previous Payment Date'] = str(Previous_Payment_Date)
        history_entry['Next Payment Due Date'] = Next_Payment_Due_Date
        history_entry['Payment Status'] = payment_status
        
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

    # Tenant Management Section
    st.subheader("👥 Tenant Management Dashboard")
    
    if not history_df.empty:
        # Calculate payment status for all records
        history_df['Payment Status'] = history_df['Next Payment Due Date'].apply(get_payment_status)
        
        # Display tenant summary table
        st.markdown("### Tenant Summary")
        
        # Create a simplified view for tenant management
        tenant_cols = ['Tenant Name', 'Telephone Number', 'City', 'Area Locality', 
                      'Predicted Rent', 'Previous Payment Date', 'Next Payment Due Date', 'Payment Status']
        
        # Filter to only columns that exist in the dataframe
        available_cols = [col for col in tenant_cols if col in history_df.columns]
        tenant_summary = history_df[available_cols].copy()
        
        # Style the dataframe based on payment status
        def color_payment_status(val):
            if val == "Overdue":
                return 'color: red; font-weight: bold'
            elif val == "Due Today":
                return 'color: orange; font-weight: bold'
            elif val == "Due Soon":
                return 'color: #FFD700; font-weight: bold'
            elif val == "Upcoming":
                return 'color: green'
            else:
                return ''
        
        # Apply styling
        styled_df = tenant_summary.style.map(color_payment_status, subset=['Payment Status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Payment status summary
        st.markdown("### Payment Status Overview")
        status_counts = tenant_summary['Payment Status'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overdue", status_counts.get("Overdue", 0))
        with col2:
            st.metric("Due Soon", status_counts.get("Due Soon", 0))
        with col3:
            st.metric("Due Today", status_counts.get("Due Today", 0))
        with col4:
            st.metric("Upcoming", status_counts.get("Upcoming", 0))
        
        # Detailed view with all data
        st.markdown("### Detailed Tenant Records")
        st.dataframe(history_df, use_container_width=True)
        
        # Delete functionality
        st.markdown("### Record Management")
        st.markdown("**Delete a row by index:**")
        delete_index = st.number_input("Row index to delete", min_value=0, max_value=len(history_df)-1, step=1)
        if st.button("🗑️ Delete Row"):
            history_df.drop(index=delete_index, inplace=True)
            history_df.reset_index(drop=True, inplace=True)
            history_df.to_csv(HISTORY_FILE, index=False)
            st.success(f"Row {delete_index} deleted successfully.")
            st.rerun()
            
        # Export functionality
        if st.button("📤 Export Tenant Data"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"tenant_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No tenant records yet. Make a prediction to get started!")

# -------------------------------
if __name__ == "__main__":
    main()
