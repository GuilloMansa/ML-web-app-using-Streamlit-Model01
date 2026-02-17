import streamlit as st
import pickle
import numpy as np
import os

MODEL_PATH = "09-Kmeans-housing-RF-(1).pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class_dict = {0: "Ingreso medio",
    1: "Ingreso medio",
    2: "Ingreso bajo",
    3: "Ingreso alto",
    4: "Ingreso alto",
    5: "Ingreso bajo"}

st.title("House Income Prediction")
st.markdown("**Powered by Guillermo Mansanta**")
st.divider()

val1 = st.slider("MedInc", 0.0, 15.0, 1.0)
val2 = st.slider("HouseAge", 0, 100, 10)
val3 = st.slider("AveRooms", 0.0, 10.0, 3.0)
val4 = st.slider("AveBedrms", 0.0, 2.0, 1.0)
val5 = st.slider("Population", 0, 5000, 1000)
val6 = st.slider("AveOccup", 0.0, 10.0, 3.0)
val7 = st.slider("Latitude", 0.0, 60.0, 30.0)
val8 = st.slider("Longitude", -150.0, 0.0, -100.0)
val9 = st.slider("MedHouseVal", 0.0, 10.0, 5.0)

if st.button("Predict"):

    features = np.array([[val1, val2, val3, val4, val5, val6, val7, val8, val9]])

    try:
        prediction = model.predict(features)[0]
        pred_class = class_dict.get(prediction, "Clase desconocida")

        st.divider()
        st.success(f"Prediction: {pred_class}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
