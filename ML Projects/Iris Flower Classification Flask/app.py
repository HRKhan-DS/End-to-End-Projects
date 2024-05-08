## app.py
from flask import Flask, render_template, request,jsonify
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and other necessary files
model_path = r'G:\3. Machine Learning-24\Iris Classification-24\data\model\best_model.pkl'
df_path = r'G:\3. Machine Learning-24\Iris Classification-24\data\cleaned_data\ready.csv'
pipeline = load(model_path)
df_train = pd.read_csv(df_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        input_data = {
            'sepal_length': float(request.form.get('sepal_length', 0.0)),
            'sepal_width': float(request.form.get('sepal_width', 0.0)),
            'petal_length': float(request.form.get('petal_length', 0.0)),
            'petal_width': float(request.form.get('petal_width', 0.0)),
            'soil_type': request.form.get('soil_type', 'default_value_if_missing')
        }
        
        print("Input Data:", input_data)  # Add this line for debugging

        # Create a DataFrame from the user input
        user_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = pipeline.predict(user_df)

        # Render the result template with the prediction variable
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return str(e), 400  # Return the error message with status code 400

@app.route('/goback')
def go_back():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)