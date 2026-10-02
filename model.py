"""Train and save the house-price prediction pipeline."""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

NUMERIC_FEATURES = ["area_sqft", "bhk", "floors", "age_years"]
CATEGORICAL_FEATURES = ["location"]
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["price"]


def train_model() -> None:
  """Fit preprocessing and regression steps, then persist them together."""
  data = pd.read_csv(DATA_PATH)
  missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
  if missing_columns:
    raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
  if data.empty:
    raise ValueError("Dataset must contain at least one training row.")

  features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
  target = data["price"]

  preprocessor = ColumnTransformer(
      transformers=[
          ("numeric", "passthrough", NUMERIC_FEATURES),
          (
              "categorical",
              OneHotEncoder(handle_unknown="ignore"),
              CATEGORICAL_FEATURES,
          ),
      ]
  )
  pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "regressor",
              RandomForestRegressor(
                  n_estimators=300,
                  min_samples_leaf=1,
                  random_state=42,
                  n_jobs=-1,
              ),
          ),
      ]
  )
  pipeline.fit(features, target)

  # The pipeline includes the fitted location encoder as well as the model.
  with MODEL_PATH.open("wb") as model_file:
    pickle.dump(pipeline, model_file)

  print(
      f"Trained on {len(data)} properties across"
      f" {data['location'].nunique()} locations."
  )
  print(f"Saved model and feature encoders to {MODEL_PATH}")


if __name__ == "__main__":
  train_model()