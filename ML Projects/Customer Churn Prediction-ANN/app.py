import streamlit as st
import pandas as pd
import joblib

# Load the trained model
rf_model = joblib.load(r'G:\Github-2025\customer_churn_ML_ANN\notebook\RandomForestModel_ov.pkl')

# Function to preprocess user input and make predictions
def predict_churn(data):
    prediction = rf_model.predict(data)
    return prediction

# Streamlit web app
def main():
    st.title('Churn Prediction')

    # Load your dataset
    df = pd.read_csv(r'G:\Github-2025\customer_churn_ML_ANN\data_set\clean_df.csv')  # Replace 'your_dataset.csv' with your dataset file path

    # Display the dataset
    st.subheader('Dataset')
    st.write(df)

    # Sidebar for user input
    st.sidebar.title('User Input')

    # Input fields for each feature (excluding 'Churn')
    user_input = {}
    for col in df.columns:
        if col != 'Churn':
            if df[col].dtype == 'object':  # Categorical features
                user_input[col] = st.sidebar.selectbox(f'Select {col}', df[col].unique())
            else:  # Numerical features
                user_input[col] = st.sidebar.number_input(f'Enter {col}', value=0)

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Button to make predictions
    if st.sidebar.button('Predict'):
        # Make predictions
        prediction = predict_churn(input_df)
        st.write('## Prediction:')
        st.write(prediction)

if __name__ == '__main__':
    main()
