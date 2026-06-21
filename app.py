import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("flight_price_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Flight Ticket Price Predictor")

st.title("✈ Flight Ticket Price Predictor")

# Dropdown values from encoders
airline = st.selectbox(
    "Airline",
    list(encoders['Airline'].classes_)
)

origin = st.selectbox(
    "Origin",
    list(encoders['Origin'].classes_)
)

destination = st.selectbox(
    "Destination",
    list(encoders['Destination'].classes_)
)

travel_class = st.selectbox(
    "Class",
    list(encoders['Class'].classes_)
)

distance = st.number_input(
    "Distance (km)",
    min_value=100,
    value=3000
)

days_before = st.slider(
    "Days Before Departure",
    min_value=1,
    max_value=120,
    value=30
)

if st.button("Predict Price"):

    airline_enc = encoders['Airline'].transform([airline])[0]
    origin_enc = encoders['Origin'].transform([origin])[0]
    destination_enc = encoders['Destination'].transform([destination])[0]
    class_enc = encoders['Class'].transform([travel_class])[0]

    data = pd.DataFrame([[
        airline_enc,
        origin_enc,
        destination_enc,
        distance,
        class_enc,
        days_before
    ]], columns=[
        'Airline',
        'Origin',
        'Destination',
        'Distance_km',
        'Class',
        'Days_Before_Departure'
    ])

    prediction = model.predict(data)[0]

    st.success(f"Predicted Ticket Price: ${prediction:.2f}")