from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

app = FastAPI()

model = mlflow.sklearn.load_model("runs:/0f99c35232f347abb31d5f1ba3d15b1f/iris_classifier_model")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}
