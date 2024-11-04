from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import pandas as pd
import mlflow
from prefect import flow
import numpy as np

mlflow.set_experiment("Test2")

@flow(log_prints=True)  
def load_data():
    iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    return iris


@flow(log_prints=True)
def preprocess():
    iris = load_data()
    features = ["sepal_length","sepal_width", "petal_length", "petal_width"]
    target = ["species"]
    return train_test_split(iris[features], iris[target].values, test_size=0.33)

@flow()
def set_params():
    return np.random.randint(10,200)

@flow(log_prints=True)
def train():
    mlflow.sklearn.autolog(registered_model_name='sgd')
    X_train, X_test, y_train, y_test = preprocess()
    print('Preprocess successed!')

    iter = set_params()
    sgd = SGDClassifier(max_iter=iter)

    with mlflow.start_run() as run:
        sgd.fit(X_train, y_train)

if __name__ == "__main__":
    train()