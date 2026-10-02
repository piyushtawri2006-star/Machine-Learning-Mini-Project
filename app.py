import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DATA_PATH = 'dataset.csv'
MODEL_PATH = 'model.pkl'

# Load valid locations from dataset
if os.path.exists(DATA_PATH):
  df = pd.read_csv(DATA_PATH)
  LOCATIONS = sorted(df['location'].unique().tolist())
else:
  LOCATIONS = ['Pune', 'Nagpur', 'Amravati', 'Wardha', 'Chandrapur']


def train_and_save_model():
  if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH) 
    X = df[['area_sqft', 'bhk', 'floors', 'age_years', 'location']]
    y = df['price']

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                'passthrough',
                ['area_sqft', 'bhk', 'floors', 'age_years'],
            ),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['location']),
        ]
    )

    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            (
                'regressor',
                RandomForestRegressor(n_estimators=100, random_state=42),
            ),
        ]
    )

    model.fit(X, y)
    with open(MODEL_PATH, 'wb') as f:
      pickle.dump(model, f)
    return model
  return None


# Load or train model on startup
if os.path.exists(MODEL_PATH):
  try:
    with open(MODEL_PATH, 'rb') as f:
      model = pickle.load(f)
  except Exception:
    model = train_and_save_model()
else:
  model = train_and_save_model()


@app.route('/')
def home():
  return render_template('index.html', locations=LOCATIONS)


@app.route('/predict', methods=['POST'])
def predict():
  try:
    bhk = float(request.form.get('bhk', 3))
    plot_size = float(request.form.get('plot_size', 1200))
    floors = float(request.form.get('floors', 2))
    age_years = float(request.form.get('age_years', 5))
    location = request.form.get('location', LOCATIONS[0])

    input_data = pd.DataFrame(
        [[plot_size, bhk, floors, age_years, location]],
        columns=['area_sqft', 'bhk', 'floors', 'age_years', 'location'],
    )

    if model is not None:
      predicted_price = float(model.predict(input_data)[0])
    else:
      predicted_price = (
          2000000
          + (bhk * 500000)
          + (plot_size * 2000)
          + (floors * 200000)
          - (age_years * 50000)
      )

    breakdown = {
        'Base Valuation': round(predicted_price * 0.40, 2),
        f'{bhk} BHK Configuration': round(predicted_price * 0.25, 2),
        f'Area Size ({plot_size} sq.ft)': round(predicted_price * 0.20, 2),
        f'Location Premium ({location})': round(predicted_price * 0.15, 2),
    }

    # Dynamic market simulation with natural dips & spikes (Volatility + CAGR)
    # Past years showing a realistic market dip 3 years ago followed by recovery
    chart_labels = [
        '3 Yr Ago',
        '2 Yr Ago',
        '1 Yr Ago',
        'Today',
        '+2 Yrs',
        '+4 Yrs',
        '+6 Yrs',
        '+8 Yrs',
        '+10 Yrs',
    ]

    # Factors incorporating past market correction and future compounding (~7.2% annual growth)
    cagr = 0.072
    factors = [
        0.78,
        0.83,
        0.91,
        1.00,
        1.00 * ((1 + cagr) ** 2),
        1.00 * ((1 + cagr) ** 4),
        1.00 * ((1 + cagr) ** 6),
        1.00 * ((1 + cagr) ** 8),
        1.00 * ((1 + cagr) ** 10),
    ]

    chart_values = [round(predicted_price * f, 2) for f in factors]

    return jsonify({
        'success': True,
        'price': round(predicted_price, 2),
        'breakdown': breakdown,
        'chart_labels': chart_labels,
        'chart_values': chart_values,
    })

  except Exception as e:
    return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
  app.run(debug=True)