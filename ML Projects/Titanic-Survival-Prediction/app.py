from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np
from scipy.stats import boxcox

app = Flask(__name__)

# Load the pre-trained model and other necessary files
model_path = r'G:\PROJECTS-2024\Titanic-ML from disaster\deploy\rf_pipeline_dep.pkl'
df_path = r'G:\PROJECTS-2024\Titanic-ML from disaster\deploy\cleaned_data.csv'
pipeline = load(model_path)
df_train = pd.read_csv(df_path)

@app.route('/')
def home():
    return render_template('index.html')  # You can create an HTML form for user input

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input_data = {
            'Pclass': request.form['Pclass'],
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'Family_size': int(request.form['Family_size']),
            'Embarked': request.form['Embarked'],
            'Fare': float(request.form['Fare']),
        }
        print("Input Data:", input_data)  # Add this line for debugging

        # Handle Box-Cox transformation for 'Fare'
        if input_data['Fare'] > 0:
            # Choose a suitable value for lambda (you can experiment with different values)
            lambda_value = 0.1
            input_data['Fare'] = ((input_data['Fare'] ** lambda_value) - 1) / lambda_value
        else:
            # If 'Fare' is 0 or negative, set it to a small positive value
            input_data['Fare'] = ((1 ** lambda_value) - 1) / lambda_value

        # Create a DataFrame from the user input
        user_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = pipeline.predict(user_df)

        # Pass the prediction result directly to the template
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)