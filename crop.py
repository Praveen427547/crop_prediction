import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("Crop_recommendation.csv", encoding="utf-8")

dt = pd.read_excel('NPK.xlsx', engine='openpyxl')


# Select features and target
X = df[["N", "P", "K", "rainfall", "humidity", "temperature"]]
y = df["label"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply feature scaling to balance all parameters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply polynomial transformation for non-linearity
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Feature Selection: Pick best 10 features based on correlation with output
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_poly, y_encoded)

# Handle class imbalance using SMOTE (Synthetic Minority Oversampling)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Train the Random Forest model with optimized hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=200,  # More trees for better accuracy
    max_depth=20,  # Deeper trees for better decision making
    min_samples_split=2,  # More splits for better granularity
    min_samples_leaf=1,  # Smaller leaf nodes for better sensitivity
    class_weight="balanced",  # Handle imbalanced data
    random_state=42
)
rf_model.fit(X_train, y_train)

# Function to fetch expected values for a given state and season
def get_expected_values(state, season):
    state_data = dt[dt["state"].str.lower() == state.lower()]
    if state_data.empty:
        raise ValueError(f"State '{state}' not found in the dataset.")

    season_map = {"zaid": "zaid", "rabi": "rabi", "kharif": "kharif"}
    if season.lower() not in season_map:
        raise ValueError(f"Season '{season}' is not valid. Choose from zaid, rabi, or kharif.")

    season_suffix = season_map[season.lower()]

    expected_temperature = state_data[f"temperature_{season_suffix}"].values[0]
    expected_humidity = state_data[f"humidity_{season_suffix}"].values[0]
    expected_rainfall = state_data[f"rainfall_{season_suffix}"].values[0]
    expected_n = state_data["N"].values[0]
    expected_p = state_data["P"].values[0]
    expected_k = state_data["K"].values[0]

    return expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall

# Prediction function with improved accuracy
def predict_crop():
    try:
        state = input("Enter state name: ").strip()
        season = input("Enter season (zaid/rabi/kharif): ").strip()

        # Fetch expected values
        expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall = get_expected_values(state, season)

        print(f"\nEnter values for prediction:")

        # User inputs (with default expected values)
        N = float(input(f"N (Nitrogen) [Expected: {expected_n}]: ") or expected_n)
        P = float(input(f"P (Phosphorus) [Expected: {expected_p}]: ") or expected_p)
        K = float(input(f"K (Potassium) [Expected: {expected_k}]: ") or expected_k)
        rainfall = float(input(f"Rainfall [Expected: {expected_rainfall}]: ") or expected_rainfall)
        humidity = float(input(f"Humidity [Expected: {expected_humidity}]: ") or expected_humidity)
        temperature = float(input(f"Temperature [Expected: {expected_temperature}]: ") or expected_temperature)

        # Normalize input values
        input_features = np.array([[N, P, K, rainfall, humidity, temperature]])
        input_features_scaled = scaler.transform(input_features)
        input_features_poly = poly.transform(input_features_scaled)
        input_features_selected = selector.transform(input_features_poly)

        # Predict using the trained model
        prediction_encoded = rf_model.predict(input_features_selected)[0]
        predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]

        print(f"\nPredicted Crop: {predicted_crop}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected err|or occurred: {e}")

# Run the interactive function
predict_crop()
