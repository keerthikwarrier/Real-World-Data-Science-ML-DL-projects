import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

# Load CSV from subfolder
iris_df = pd.read_csv("data/iris.csv")

X = iris_df.drop("Species", axis =1)
y = iris_df["Species"]

label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)
y_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25 , random_state =21)

example_input = pd.DataFrame([X_train.iloc[0]])

# Start MLflow experiment
mlflow.set_experiment("Iris_Classifier_Experiment")

# Track with MLflow
with mlflow.start_run(run_name="IrisClassification_KNNeighborClassifier"):
    knn_model = KNeighborsClassifier(n_neighbors=4)  # n_neighbors is the K value and by default, it's value is equal to 5
    knn_model.fit(X_train, y_train)  # suprvised learning

    y_test_pred_knn = knn_model.predict(X_test)
    y_train_pred_knn = knn_model.predict(X_train)
    accuracy=accuracy_score(y_test, y_test_pred_knn)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(knn_model, name="iris_classifier_model",input_example=example_input)

    print("Logged model with accuracy:", accuracy)

