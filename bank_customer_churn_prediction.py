import streamlit as st
import pandas as pd
import joblib

# set page config
st.set_page_config(
    page_title="Churn Customer Prediction", 
    page_icon="🏦", 
    layout="centered",
    initial_sidebar_state="expanded"
)


# load the model
@st.cache_resource
def load_model_arfifact():
    try:
        model_pipeline = joblib.load("churned_model_prediction.pkl")
        feature_names = joblib.load("churn_feature_columns.pkl")
        return model_pipeline, feature_names
    except FileNotFoundError as e:
        st.error(f"Model artifact not found: {e}")
        st.error("Please ensure churned_model_prediction.pkl and churn_feature_columns.pkl existsinside directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
def predict_churn(customer_data, model_pipeline):
    try:
        customer_df = pd.DataFrame([customer_data])
        
        
        # make prediction
        prediction = model_pipeline.predict(customer_df)[0]
        probability = model_pipeline.predict_proba(customer_df)[0]
        
        # churn risk level
        churn_risk = probability[1]
        if churn_risk >= 0.7:
            risk_level = "High",
            risk_color = "🔴"
        elif churn_risk >= 0.4:
            risk_level = "Medium"
            risk_color = "🟡"
        else:
            risk_level = "Low"
            risk_color = "🟢"
        
        result = {
            "prediction": "Churned" if prediction == 1 else "Retained",
            "churn_probability": churn_risk,
            "retained_probability": probability[0],
            "risk_level": risk_level,
            "risk_color": risk_color
        }
        
        return result
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()
    
    
def main():
    # header
    st.header("Customer Churn Prediction")
    st.write("Predict whether a bank customer is likely to churn based on their profile and behavior.")
    
    # load model
    with st.spinner("Loading model..."):
        model_pipeline, feature_names = load_model_arfifact()
    
    # success message
    st.success("✅Model loaded successfully!")
    
    st.markdown("---")
    st.subheader("Enter Customer Details")
    
    # create two columns for input
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Personal Information")
        credit_score = st.number_input(
            "Credit Score",
            min_value=350, 
            max_value=850, 
            value=650,
            help="Customer Credit Score (350-850)"
        )
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=92,
            value=30,   
            help="Customer's age  by uear (18-92)"
        )
        
        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=1,    
            help="Number of years the customer has been with the bank (0-10)"
        )   
    

if __name__ == "__main__":
    main()