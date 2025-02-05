
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('Crop_recommendation.xlsx', engine='openpyxl')
dt = pd.read_excel('NPK.xlsx', engine='openpyxl')


X = df[["N", "P", "K", "rainfall", "humidity", "temperature"]]
y = df["label"]

# Encoding target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Decision Tree model
dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_model.fit(X_train, y_train)


def get_expected_values(state, season):
    # Filter the dataset for the specified state
    state_data = dt[dt["state"].str.lower() == state.lower()]

    if state_data.empty:
        raise ValueError(f"State '{state}' not found in the dataset.")

    # Map season to column suffix
    season_map = {
        "zaid": "zaid",
        "rabi": "rabi",
        "kharif": "kharif"
    }

    if season.lower() not in season_map:
        raise ValueError(f"Season '{season}' is not valid. Choose from zaid, rabi, or kharif.")

    season_suffix = season_map[season.lower()]

    # Extract expected values
    expected_temperature = state_data[f"temperature_{season_suffix}"].values[0]
    expected_humidity = state_data[f"humidity_{season_suffix}"].values[0]
    expected_rainfall = state_data[f"rainfall_{season_suffix}"].values[0]
    expected_n = state_data["N"].values[0]
    expected_p = state_data["P"].values[0]
    expected_k = state_data["K"].values[0]

    return expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall

def predict_crop_decision_tree():
    try:
        # Get state input
        state = input("Enter state name: ").strip()

        # Get season input
        season = input("Enter season (zaid/rabi/kharif): ").strip()

        # Fetch expected values
        expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall = get_expected_values(state, season)

        print(f"\nEnter values for prediction:")

        # Get feature inputs
        N = float(input(f"N (Nitrogen) [Expected: {expected_n}]: ") or expected_n)
        P = float(input(f"P (Phosphorus) [Expected: {expected_p}]: ") or expected_p)
        K = float(input(f"K (Potassium) [Expected: {expected_k}]: ") or expected_k)
        rainfall = float(input(f"Rainfall [Expected: {expected_rainfall}]: ") or expected_rainfall)
        humidity = float(input(f"Humidity [Expected: {expected_humidity}]: ") or expected_humidity)
        temperature = float(input(f"Temperature [Expected: {expected_temperature}]: ") or expected_temperature)

        # Create a feature array with the exact columns used during training
        input_features = [[N, P, K, rainfall, humidity, temperature]]

        # Predict using the trained model
        prediction_encoded = dt_model.predict(input_features)[0]
        predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]

        print(f"\nPredicted Crop: {predicted_crop}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the interactive function1
predict_crop_decision_tree()

