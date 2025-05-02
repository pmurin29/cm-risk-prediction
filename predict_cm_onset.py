
import pandas as pd
import joblib

# Load the trained model and imputer
model = joblib.load('random_forest_model.joblib')
imputer = joblib.load('imputer.joblib')

# Define the feature columns (11 features used during training)
trained_features = [
    'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8',
    'Day 9', 'Day 10', 'Day 11'
]

def predict_cm_onset(new_data: pd.DataFrame):
    # Ensure the new data only contains the required features
    new_data = new_data[trained_features]
    
    # Impute any missing values in the new data
    new_data_imputed = imputer.transform(new_data)

    # Make predictions using the trained model
    predicted_day_cm = model.predict(new_data_imputed)

    return predicted_day_cm[0]  # Return the predicted day of CM onset

# Example usage
new_data = pd.DataFrame({
    'Day 1': [0.5],
    'Day 2': [1.2],
    'Day 3': [2.3],
    'Day 4': [1.0],
    'Day 5': [0.8],
    'Day 6': [1.1],
    'Day 7': [0.9],
    'Day 8': [1.3],
    'Day 9': [1.4],
    'Day 10': [0.7],
    'Day 11': [1.2]
})

predicted_day = predict_cm_onset(new_data)
print(f"Predicted Day of CM Onset: {predicted_day}")
