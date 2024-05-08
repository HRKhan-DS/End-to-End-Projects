import pandas as pd
from joblib import load

# Load the trained model
model_path = r'G:\PROJECTS-2024\Titanic-ML from disaster\data\model\rf_pipeline.pkl'
pipeline = load(model_path)

# Load the DataFrame from the CSV file
csv_path = r"G:\PROJECTS-2024\Titanic-ML from disaster\data\preprocessed_data\cleaned_train_data.csv"
df_train = pd.read_csv(csv_path)

print(df_train.head())

def predict_survival(passenger_id, model, df):
    passenger_data = df[df['PassengerId'] == passenger_id]

    if not passenger_data.empty:
        # Extract relevant features for prediction
        features = passenger_data.drop('Survived', axis=1)

        # Make prediction
        prediction = pipeline.predict(features)

        # Retrieve the actual survival status
        actual_survival_status = passenger_data['Survived'].values[0]

        return (f"Passenger {passenger_id} is predicted to {'survive' if prediction[0] == 1 else 'not survive'}. "
                f"Actual survival status: {'Survived' if actual_survival_status == 1 else 'Not Survived'}")

    return f"No data found for Passenger {passenger_id}."

# Example usage
passenger_id_to_predict =2
result = predict_survival(passenger_id_to_predict, pipeline, df_train)
print(result)