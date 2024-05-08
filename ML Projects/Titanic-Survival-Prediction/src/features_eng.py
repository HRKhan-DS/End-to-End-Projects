#### src/features_eng.py
from sklearn.preprocessing import LabelEncoder
from src.load_data import load_data
from src.preprocess import preprocess_data

def features_engineering(data):

    # Add feature: Family Size
    data['Family_size'] = data['SibSp'] + data['Parch']

    # Drop unnecessary columns
    data.drop(columns=['SibSp', 'Parch'], inplace=True)

    return data

# Example usage:
engaged_data = features_engineering(preprocess_data(load_data()))
print(engaged_data.head())