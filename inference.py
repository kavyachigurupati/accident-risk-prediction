import mlflow.lightgbm
import polars as pl
from polars import selectors as cs

# Path to the folder containing MLmodel and model.lgb
model_path = "mlruns/0/models/m-1e36f97cc27145f8bd98740893d8142c/artifacts"

model = mlflow.lightgbm.load_model(model_path)
# Load test CSV
test = pl.read_csv("playground-series-s5e10/test.csv")

# Convert string/boolean columns like you did for training
string_cols = test.select(pl.col(pl.Utf8)).columns

test = test.with_columns(
    pl.col(string_cols).cast(pl.Categorical).to_physical(),
    pl.col(pl.Boolean).cast(pl.Int8)
)

# Drop target column if it exists in the test set
# X_test = test.drop("accident_risk", ignore_errors=True)
read_test_data = pl.read_csv("playground-series-s5e10/test.csv")
test = pl.DataFrame(read_test_data)
print(test)

X_test = test.with_columns(
    cs.string().cast(pl.Categorical).to_physical(),
    cs.boolean().cast(pl.Int8)
)

y_pred = model.predict(X_test)

print(y_pred[:10])  # first 10 predictions

test = test.with_columns(pl.Series("prediction", y_pred))

test.write_csv("test_with_predictions.csv")

print("Predictions saved to test_with_predictions.csv")