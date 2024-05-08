### deploy/deploy_train_model.py
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load

path = r"G:\PROJECTS-2024\Titanic-ML from disaster\data\preprocessed_data\cleaned_train_data.csv"
deploy_df = pd.read_csv(path)

deploy_df.drop(columns=['PassengerId'], inplace=True)

# Save the cleaned data to a CSV file
cleaned_data_path = "G:\PROJECTS-2024\Titanic-ML from disaster\deploy\cleaned_data.csv"
deploy_df.to_csv(cleaned_data_path, index=False)
print(f"Cleaned data saved to {cleaned_data_path}")

# Assume 'Survived' is the target column
X = deploy_df.drop(columns=['Survived'])
y = deploy_df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessor (make sure it's consistent with preprocessing in preprocess.py)
numeric_features = ['Age', 'Family_size','Fare']
ordinal_features = ['Pclass']
categorical_features = ['Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("ord", OrdinalEncoder(), ordinal_features),
        ("cat", OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and RandomForestClassifier
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_depth= None, min_samples_leaf= 2, min_samples_split= 5, n_estimators=50))
])

rf_pipeline.fit(X_train, y_train)

y_pred_train_rf = rf_pipeline.predict(X_train)
y_pred_test_rf = rf_pipeline.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming y_pred_train and y_pred_test are your predicted values
accuracy_train_rf = accuracy_score(y_train, y_pred_train_rf)
accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)

print(f"Accuracy_train_RF: {accuracy_train_rf}\nAccuracy_test_RF: {accuracy_test_rf}")

cm = confusion_matrix(y_test, y_pred_test_rf)
print(f"Confusion_matrix:\n {cm}")

# Rename the variable to avoid conflicts with the function name
classification_report_rf = classification_report(y_test, y_pred_test_rf)

print(f"Classification_report:\n{classification_report_rf}")


# Save the trained model to a file
model_output_path = r'G:\PROJECTS-2024\Titanic-ML from disaster\deploy\rf_pipeline_dep.pkl'
dump(rf_pipeline, model_output_path)
print(f"Model saved to {model_output_path}")