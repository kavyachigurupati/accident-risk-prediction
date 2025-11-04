from fastapi import FastAPI
import mlflow.lightgbm
import pandas as pd
from pydantic import BaseModel
import polars as pl
from polars import selectors as cs

model_path = "mlruns/0/models/m-1e36f97cc27145f8bd98740893d8142c/artifacts"
# Load your ML model

model = mlflow.lightgbm.load_model(model_path)

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    id: int
    road_type: str
    num_lanes: int
    curvature: float
    speed_limit: int
    lighting: str
    weather: str
    road_signs_present: bool
    public_road: bool
    time_of_day: str
    holiday: bool
    school_season: bool
    num_reported_accidents: int
    # add all features your model expects

@app.post("/predict")
def predict(data: InputData):
    df = preprocess_input(data)
    # LightGBM expects NumPy array
    y_pred = model.predict(df.to_numpy())
    return {"prediction": float(y_pred[0])}

def preprocess_input(data: InputData):
    # Convert single input to Polars DataFrame
    df = pl.DataFrame([data.dict()])

    # Cast string columns to Categorical and booleans to Int8
    string_cols = df.select(pl.col(pl.Utf8)).columns
    df = df.with_columns(
        pl.col(string_cols).cast(pl.Categorical).to_physical(),
        pl.col(pl.Boolean).cast(pl.Int8)
    )
    return df

