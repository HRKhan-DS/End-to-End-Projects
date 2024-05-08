## setup.py

from setuptools import setup, find_packages

setup(
    name='customer_churn_project_ML_ANN',
    version='0.1',
    packages=find_packages(),
    author='Md. Harun-Or-Rashid Khan',
    author_email='mdhrkhandata.analyst@gmail.com',
    description='A project for predicting customer churn with machine learning algorithm and artificial Neural Network',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'missingno',
        'scipy',
        'xgboost',
        'keras',
        'tansorflow',
        'scikit-learn',
    ],
)