import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("accident_severity_model.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_order = joblib.load("feature_order.pkl")

# Feature mappings for human-friendly UI
day_of_week_map = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
}

urban_rural_map = {
    "Urban": 1, "Rural": 2
}

severity_labels = {
    1: "Fatal", 2: "Severe", 3: "Slight"
}

# Rename for clarity
pedestrian_facility_options = [
    "Zebra crossing",
    "Pedestrian phase at traffic signal junction",
    "No crossing within 50 meters",
    "Pedestrian island (formerly 'Central refuge')",
    "Non-junction pedestrian crossing",
    "Footbridge or subway"
]

# Streamlit UI
st.title("Accident Severity Prediction")

weather = st.selectbox("Weather Conditions", [
    "Fine without high winds",
    "Fine with high winds",
    "Raining without high winds",
    "Raining with high winds",
    "Snowing without high winds",
    "Snowing with high winds",
    "Fog or mist"
])

road_surface = st.selectbox("Road Surface Conditions", [
    "Dry", "Wet/Damp", "Frost/Ice", "Snow", "Flood (Over 3cm of water)", "Normal"
])

light_conditions = st.selectbox("Light Conditions", [
    "Daylight: Street light present",
    "Darkness: Street lights present and lit",
    "Darkness: Street lights present but unlit",
    "Darkness: Street lighting unknown",
    "Darkeness: No street lighting"
])

urban_or_rural = st.selectbox("Urban or Rural Area", list(urban_rural_map.keys()))

road_type = st.selectbox("Road Type", [
    "Single carriageway", "Dual carriageway", "One way street",
    "Roundabout", "Slip road", "Unknown"
])

speed_limit = st.selectbox("Speed Limit (mph)", [10, 15, 20, 30, 40, 50, 60, 70])

num_vehicles = st.selectbox("Number of Vehicles", ["1", "2", "3", "4", "5+"])

num_casualties = st.selectbox("Number of Casualties", ["1", "2", "3", "4", "5", "6", "7", "8+"])

day_of_week = st.selectbox("Day of Week", list(day_of_week_map.keys()))

junction_control = st.selectbox("Junction Control", [
    "Automatic traffic signal", "Giveway or uncontrolled", "Stop Sign", "Authorised person"
])

pedestrian_facility = st.selectbox("Pedestrian Crossing Facility", pedestrian_facility_options)

# Prepare data
input_data = {
    "Weather_Conditions": weather,
    "Road_Surface_Conditions": road_surface,
    "Light_Conditions": light_conditions,
    "Urban_or_Rural_Area": urban_rural_map[urban_or_rural],
    "Road_Type": road_type,
    "Speed_limit": int(speed_limit),
    "Number_of_Vehicles": num_vehicles,
    "Number_of_Casualties": num_casualties,
    "Day_of_Week": day_of_week_map[day_of_week],
    "Junction_Control": junction_control,
    "Pedestrian_Crossing-Physical_Facilities": pedestrian_facility
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical features using saved encoders
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Ensure feature order matches training
input_df = input_df[feature_order]

# Predict
if st.button("Predict Severity"):
    prediction = model.predict(input_df)[0]
    severity = severity_labels.get(prediction, "Unknown")
    st.success(f"Predicted Accident Severity: {severity}")
