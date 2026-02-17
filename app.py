from pickle import load
import streamlit as st


model = load(open('09-Kmeans-housing-RF-(1).pkl', 'rb'))
class_dict = {"0": "Ingreso medio",
              "1": "Ingreso medio",
              "3": "Ingreso alto",
              "4": "Ingreso alto",
              "2": "Ingreso bajo",
              "5": "Ingreso bajo"}

st.title("House/ingreso - Model prediction")
st.markdown("""Power by: [Guillermo Mansanta]""")
st.divider()

val1 = st.slider("MedInc", min_value=0, max_value=15, step=0.1)
val2 = st.slider("HouseAge", min_value=0, max_value=100, step=1)
val3 = st.slider("AveRooms", min_value=0, max_value=10, step=1)
val4 = st.slider("AveBedrms", min_value=0, max_value=2, step=0.1)
val5 = st.slider("Population", min_value=0, max_value=5000, step=1)
val6 = st.slider("AveOccup", min_value=0, max_value=10, step=0.1)
val7 = st.slider("Latitude", min_value=0, max_value=60, step=0.1)
val8 = st.slider("Longitude", min_value=-150, max_value=0, step=0.1)
val9 = st.slider("MedHouseVal", min_value=0, max_value=10, step=0.1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9]])[0])
    pred_class = class_dict[prediction]
    st.divider()
    st.write("Prediction:", pred_class)
    st.divider()
