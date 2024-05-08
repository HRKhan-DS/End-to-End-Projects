##### src/preprocess.py
import pandas as pd
from src.load_data import load_data
from src.config import raw_file

def preprocess_data(data):  # Accept the data as an argument
    # Handle missing values (customize based on your data)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

    # Map Pclass values to "first", "second", and "third"
    class_mapping = {1: 'first', 2: 'second', 3: 'third'}
    data['Pclass'] = data['Pclass'].map(class_mapping)

    # Drop specified columns
    data.drop(columns=["Name", "Cabin", "Ticket"], inplace=True)

    return data

# Example usage:
if __name__ == "__main__":
    preprocessed_data = preprocess_data(load_data())
    print(preprocessed_data.head())
