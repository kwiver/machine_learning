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
            help="Customer credit score (350-850)"
        )
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=92,
            value=30,   
            help="Customer's age  by year (18-92)"
        )
        
        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=1,    
            help="Number of years the customer has been with the bank (0-10)"
        )
        
        geography = st.selectbox(
            "Geography",
            options=["France", "Spain", "Germany"],
            help="Customer's country of residence"
        )
        
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Customer's gender"
        )
        
        
    with col2:
        st.markdown("Account Information")
        balance = st.number_input(
            "Account Balance (₦)",
            min_value=0.0,
            max_value=250_898.09,
            value=10_000.00,
            help="Customer's account balance"
        )
        
        num_of_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2,    
            help="Number of bank products the customer is using (1-4)"
        )
        
        estimated_salary = st.number_input(
            "Estimated Salary (₦)",
            min_value=11.58,
            max_value=199_992.48,
            value=1000.00,
            help="Customer's estimated annual salary"
        )
        
        has_credit_card = st.selectbox(
            "Has Credit Card",
            options=["Yes", "No"],
            help="Whether the customer has a credit card"
        )
        
        is_active_member = st.selectbox(
            "Is Active Member",
            options=["Yes", "No"],
            help="Whether the customer is an active member"
        )
    
    service_rating = st.slider(
        "Service Rating",
        min_value=1,
        max_value=5,
        value=3,
        help="Customer's rating of the bank's service (1-5)"
    )
    
    
    st.markdown("---")
    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        customer_data = {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Geography": geography,
            "Gender": gender,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "EstimatedSalary": estimated_salary,
            "HasCrCard": has_credit_card,
            "IsActiveMember": is_active_member,
            "ServiceRating": service_rating
        }
        
        # make prediction
        with st.spinner("Analyzing customer data..."):
            result = predict_churn(customer_data, model_pipeline)
            
            if result:
                st.markdown("---")
                st.subheader("📊Prediction Analysis")
            
                if result["prediction"] == "Churned":
                    st.warning(f"### ⚠️ Prediction: Customer is likely to CHURN ⚠️.")
                else:
                    st.success(f"### ✅ Prediction: Customer is likely to be RETAINED ✅.")
                
                # risk_level
                st.write(f"***Risk color: {result['risk_color']} | Risk level: {result['risk_level']}***")
                
                # check probability
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Churn Probability",
                        value=f"{result["churn_probability"]:.2%}",
                    )
                with col2:
                    st.metric(
                        "Retained Probability",
                        value=f"{result["retained_probability"]:.2%}",
                    )
            
                # interpretation
                st.markdown("---")
                st.subheader("🔍 Interpretation")
                
                if result["risk_level"] == "High":
                    st.warning("""
                        **High Risk Customer**
                        - Immediate action required
                        - Consider retention offers
                        - Personalized engagement needed
                        - Monitor closely  
                    """)
                elif result["risk_level"] == "Medium":
                    st.info("""
                        **Medium Risk Customer**
                        - Monitor customer activity
                        - Engage with targeted campaigns
                        - Improve service quality
                        - Regular follow-ups
                    """)
                else:
                    st.success("""
                        **Low Risk Customer**
                        - Customer is satisfied
                        - Maintain current service standards
                        - Continue regular engagement
                        - Opportnity for upselling
                    """)
                    
                # summary
                st.markdown("---")
                st.subheader("📋 Customer Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.write(f"***Credit Score:*** {credit_score}")
                    st.write(f"***Age:*** {age} years")
                    st.write(f"***Tenure:*** {tenure} years")
                    st.write(f"***Geography:*** {geography}")
                    st.write(f"***Gender:*** {gender}")
                with summary_col2:
                    st.write(f"***Accont Balance:*** ₦{balance:,.2f}")
                    st.write(f"***Number of Products:*** {num_of_products}")
                    st.write(f"***Estimated Salary:*** ₦{estimated_salary:,.2f}")
                    st.write(f"***Has Credit Card:*** {has_credit_card}")
                    st.write(f"***Is Active Member:*** {is_active_member}")
                    
                st.write(f"***Service Rating:*** {service_rating} / 5")
                
                # recommendations
                st.markdown("---")
                st.subheader("💡 Recommendations")
                
                if result["churn_probability"] > 0.7:
                    st.markdown("""
                        1. **Immediate Contact**: Reach out to customer within 24 hours.
                        2. **Special Offers**: Provide exclusive discounts or offers to retain the customer.
                        3. **Service Review**: Investigate any service issues
                        4. **Account Manager**: Assign a dedicated account manager to the customer.
                        5. **Feedback Collection**: Conduct satisfaction survey
                    """)
                elif result["churn_probability"] > 0.4:
                    st.markdown("""
                        1. **Engagement**: Increase communication frequency.
                        2. **Product Recommendations**: Suggest relevant products.
                        3. **Service Quality Improvement**: Address any service issues proactively.
                        4. **Regular Follow-ups**: Schedule quarterly reviews.
                    """)
                else:
                    st.markdown("""
                        1. **Maintain Relationship**: Continue current engagement and service level.
                        2. **Upsell Opportunities**: Introduce premium products or services.
                        3. **Referral Program**: Encourage customers to refer new ones.
                        4. **Feedback**: Collect Positive feedback and testimonials for marketing purposes.
                        5. **Recognition**: Acknowledge loyalty and satisfation.
                    """)
    # footer
    st.markdown("---")
    st.caption("Customer churn prediction system")
    
        
        
if __name__ == "__main__":
    main()