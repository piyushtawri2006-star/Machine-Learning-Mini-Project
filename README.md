# House Price Prediction

A small Flask app that estimates property prices from a synthetic training dataset. The fitted scikit-learn pipeline includes both the Random Forest and its location encoder, saved together in `model.pkl`.

## Run locally

Use Python 3.9 or newer. From this folder, install dependencies and train the model:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python model.py
python app.py
```

Open http://127.0.0.1:5000 in your browser. Run `python model.py` again whenever `dataset.csv` changes. The dataset is synthetic and the returned values are illustrative, not a real estate appraisal.

## Prediction API

`POST /predict` accepts form fields `bhk`, `plot_size`, `floors`, `age_years`, and `location`. It returns a JSON model estimate and an input-based planning range.
