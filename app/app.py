import streamlit as st
import requests

st.title("Iris Species Prediction 🌼")
st.write("Enter the following Iris flower measurements to predict the species:")

# Define a dictionary mapping species to emojis
species_emojis = {
    "setosa": ":cherry_blossom: Iris-setosa",
    "versicolor": ":tulip: Iris-versicolor",
    "virginica": ":hibiscus: Iris-virginica"
}

with st.form("iris_form"):
    sepal_length = st.text_input("Sepal Length (cm)")
    sepal_width = st.text_input("Sepal Width (cm)")
    petal_length = st.text_input("Petal Length (cm)")
    petal_width = st.text_input("Petal Width (cm)")

    submit = st.form_submit_button("Predict Iris Species")

def get_api(params):
    url = "http://api:8086/predict/"
    response = requests.get(url, params=params)
    return response.json().get('prediction', "Error: No prediction returned")

if submit:
    try:
        # Convert inputs to float and validate
        params = {
            "sepal_length": float(sepal_length),
            "sepal_width": float(sepal_width),
            "petal_length": float(petal_length),
            "petal_width": float(petal_width)
        }
        # Fetch prediction
        prediction = get_api(params)
        
        # Get emoji for the prediction, default to the prediction if not found
        prediction_with_emoji = species_emojis.get(prediction, prediction)

        st.header("Prediction Result")
        st.write(f"The predicted Iris species is: {prediction_with_emoji}")

    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
