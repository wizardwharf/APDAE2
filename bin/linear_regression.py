import numpy as np
from pickle import load, dump
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = load_diabetes()
X = data.data
y = data.target

def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def pickle_model(model, model_file):
    with open(model_file, 'wb') as f:
        dump(model, f)

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = load(f)
    return model

def get_regression_params(model):
    return {'intercept':model.intercept_, 'coefficients':model.coef_}
