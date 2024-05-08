#### src/outlier_handling.py
import numpy as np
from scipy.stats import boxcox
from src.load_data import load_data
from src.features_eng import features_engineering
from src.preprocess import preprocess_data

def outlier_handling(data):
    Q1 = data['Fare'].quantile(0.25)
    Q3 = data['Fare'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data['Fare'] = np.where(data['Fare'] > upper, upper, data['Fare'])

    # Apply the Box-Cox transformation
    data['Fare'] = boxcox(data['Fare'] + 1)[0]  # Adding 1 to handle zero values

    return data

# Example usage in outlier_handling.py:
cleaned_train_data = outlier_handling(features_engineering(preprocess_data(load_data())))

print(cleaned_train_data.head())

# Save the cleaned data to a CSV file
path = r'G:\PROJECTS-2024\Titanic-ML from disaster\data\preprocessed_data\cleaned_train_data.csv'
cleaned_train_data.to_csv(path, index=False)

print("Cleaned data saved successfully.")