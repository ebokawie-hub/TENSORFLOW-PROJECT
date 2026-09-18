
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon=":robot:")

st.title("MNIST Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9) to get a prediction!")

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_model.h5')
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L") # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image for the model
    img_array = np.array(image)
    img_resized = tf.image.resize(np.expand_dims(img_array, -1), (28, 28))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0) # Add batch dimension

    # Make prediction
    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: Digit {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

st.write("---")
st.write("To run this Streamlit app locally, save the `app.py` and `mnist_model.h5` files in the same directory, then run `streamlit run app.py` in your terminal.")
