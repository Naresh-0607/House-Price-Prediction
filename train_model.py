import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
FILTERED_PATH = os.path.join(BASE_DIR, "filtered_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def prepare_data() -> pd.DataFrame:
    """Apply the same filtering logic used in the notebook."""
    house_data = pd.read_csv(DATA_PATH)
    house_filtered = house_data[
        (house_data["price"] < 150000) & (house_data["price"] != 0)
    ][["price", "sqft_living"]]
    house_filtered.to_csv(FILTERED_PATH, index=False)
    return house_filtered


def train_and_export_model() -> None:
    df = prepare_data()
    model = LinearRegression()
    model.fit(df[["sqft_living"]], df["price"])
    joblib.dump(model, MODEL_PATH)
    print(f"Model exported successfully to: {MODEL_PATH}")
    print(f"Coefficient: {model.coef_[0]:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")


if __name__ == "__main__":
    train_and_export_model()