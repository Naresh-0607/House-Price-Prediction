# House Price Prediction (Linear Regression + Flask)

A simple machine learning project that predicts house price from square feet (`sqft_living`) using Linear Regression, then serves predictions through a Flask web UI.

## Features

- Data filtering and preprocessing from `data.csv`
- Linear Regression model training with scikit-learn
- Exported model file (`model.pkl`) using joblib
- Flask UI to enter square feet and get predicted price

## Project Structure

- `data.csv` - Original dataset
- `filtered_data.csv` - Filtered dataset used for training
- `linear.ipynb` - Notebook used for exploration and initial model building
- `train_model.py` - Script to train and export the model
- `model.pkl` - Exported trained model
- `app.py` - Flask application
- `templates/index.html` - Frontend UI template
- `requirements.txt` - Python dependencies

## Model Logic

The training script applies the same filtering logic used in the notebook:

- Keep rows where `price < 150000`
- Exclude rows where `price == 0`
- Use `sqft_living` as input feature
- Use `price` as target

The model equation is:

$$
\hat{y} = m x + b
$$

Where:

- $x$ = square feet (`sqft_living`)
- $\hat{y}$ = predicted house price

## Requirements

- Python 3.10+ (recommended)
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Train and export model:

```bash
python train_model.py
```

2. Start Flask app:

```bash
python app.py
```

3. Open in browser:

- http://127.0.0.1:5000

## Usage

- Enter a square feet value in the input box
- Click **Predict Price**
- The app shows the predicted house price

## Notes

- This project currently uses one feature (`sqft_living`) for prediction.
- Flask runs in development mode by default.
- The prediction quality depends on data quality and filtering assumptions.

## Future Improvements

- Add model evaluation metrics (R2, MAE, RMSE)
- Add multiple features (bedrooms, bathrooms, location)
- Add API endpoint for JSON predictions
- Add Docker support for easy deployment
