import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Load dataset (assuming df is available)
df = pd.read_excel('Crop_recommendation.xlsx', engine='openpyxl')  
dt = pd.read_excel('NPK.xlsx', engine='openpyxl')

# Feature selection
X = df[["N", "P", "K", "rainfall", "humidity", "temperature"]]
y = df["label"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_poly, y_encoded)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y_encoded)

# Train model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_balanced, y_balanced)


def get_expected_values(state, season):
    state_data = dt[dt["state"].str.lower() == state.lower()]
    if state_data.empty:
        return None
    
    season_map = {"zaid": "zaid", "rabi": "rabi", "kharif": "kharif"}
    if season.lower() not in season_map:
        return None
    
    season_suffix = season_map[season.lower()]
    return {
        "N": state_data["N"].values[0],
        "P": state_data["P"].values[0],
        "K": state_data["K"].values[0],
        "temperature": state_data[f"temperature_{season_suffix}"].values[0],
        "humidity": state_data[f"humidity_{season_suffix}"].values[0],
        "rainfall": state_data[f"rainfall_{season_suffix}"].values[0]
    }

# Function to set background image from URL
def set_background_from_url(image_url):
    # Fetch image from URL
    response = requests.get(image_url)
    if response.status_code == 200:
        # Encode the image to base64
        encoded = base64.b64encode(response.content).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    else:
        st.error("Failed to load the image from the URL")

# Mapping of crops to their respective background image URLs
crop_images = {
    "rice": "https://t4.ftcdn.net/jpg/09/47/19/71/360_F_947197189_OmyKmvXf25RlHFODviXKNL1zddUMFIaN.jpg",  # Replace with your rice image URL
    "coconut": "https://parachutekalpavriksha.org/cdn/shop/articles/Sure_ways_to_keep_the_coconut_tree_healthy.jpg",  # Replace with your wheat image URL
    "apple": "https://usapple.org/wp-content/uploads/2019/11/feature-image-chlorpyrifos.jpg",  # Replace with your corn image URL
    "watermelon" : "https://i.ytimg.com/vi/9D2WgDKDTy4/hq720.jpg",
    "muskmelon" : "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMXXG-GCCjjh3ERJVAYkBAe0H-wAb3pXwXqQ&s.jpg",
    "maize" : "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpcPJn8xeSS5N4LBzA9JZv0TFkXJSDPxVIHQ&s.jpg",
    "orange" : "https://cdn.freshfruitportal.com/2021/07/WH_2008_PC08_0754_rgb-1024x683.jpg",
    "cotton" : "https://agrihunt.com/wp-content/uploads/2015/11/022309-Cotton-Center-460-284-PG-1.jpg",
    # Add other crops and their URLs here
}

# Streamlit UI
st.title("Crop Recommendation System")

# Get unique states from dataset
state_list = sorted(dt["state"].dropna().unique())

# State selection using dropdown
state = st.selectbox("Select State:", state_list)

# Season selection using dropdown
season = st.selectbox("Select Season:", ["zaid", "rabi", "kharif"])

if state and season:
    expected_values = get_expected_values(state, season)
    if expected_values:
        N = st.number_input("Nitrogen (N)", value=expected_values["N"])
        P = st.number_input("Phosphorus (P)", value=expected_values["P"])
        K = st.number_input("Potassium (K)", value=expected_values["K"])
        rainfall = st.number_input("Rainfall", value=expected_values["rainfall"])
        humidity = st.number_input("Humidity", value=expected_values["humidity"])
        temperature = st.number_input("Temperature", value=expected_values["temperature"])
        
        if st.button("Predict Crop"):
            input_features = np.array([[N, P, K, rainfall, humidity, temperature]])
            input_features_scaled = scaler.transform(input_features)
            input_features_poly = poly.transform(input_features_scaled)
            input_features_selected = selector.transform(input_features_poly)
            
            prediction_encoded = rf_model.predict(input_features_selected)[0]
            predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]
            
            st.success(f"Predicted Crop: {predicted_crop}")

            # If the predicted crop exists in the dictionary, set the background image
            if predicted_crop.lower() in crop_images:
                set_background_from_url(crop_images[predicted_crop.lower()])
            else:
                st.error("No image available for the predicted crop.")
    else:
        st.error("Invalid state or season. Please try again.")
