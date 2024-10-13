import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model("Copy of breast_cancer_classifier.h5")


# Function to make predictions
def classify_image(img):
    img = img.resize((224, 224))  # Resize the image to the input size of the model
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    return predictions


# Streamlit app title
st.title("Breast_Cancer_Classification")

# Upload image section
uploaded_file = st.file_uploader("Upload an image of a breast cancer tumor", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    st.write("Classifying The Image...")
    prediction = classify_image(image)

    # Display prediction
    class_names = ["Benign", "Malignant", "Normal", "Unknown"]
    st.write(f"Prediction: {class_names[np.argmax(prediction)]}")
