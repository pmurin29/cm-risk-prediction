# cm-risk-prediction
# CM Risk Prediction Model

This project provides a machine learning model for predicting the onset of experimentla cerebral malaria (ECM) in mice infected with Plsamodium berghei ANKA based on parasitemia data.

## Files
- `predict_cm_onset.py`: The Python script that loads the trained model and makes predictions.
- `random_forest_model.joblib`: The trained Random Forest model.
- `imputer.joblib`: The imputer used for handling missing values in input data.

## Requirements
The following Python packages are required:
- `joblib`
- `pandas`
- `scikit-learn`

You can install the required packages by running:
```bash
pip install -r requirements.txt
