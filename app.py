from fastapi import FastAPI
import joblib
import numpy as np
from schemas import IrisInput
from sklearn.datasets import load_iris
import os

app = FastAPI(title="Iris ML API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_pipeline.pkl")

model = joblib.load(MODEL_PATH)
target_names = load_iris().target_names

@app.get("/")
def home():
    return {"message": "Iris ML API is running 🚀"}

@app.post("/predict")
def predict(data: IrisInput):
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(input_data)
    class_name = target_names[prediction[0]]

    return {
        "prediction": int(prediction[0]),
        "class_name": class_name
    }
