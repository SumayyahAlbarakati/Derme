# prompt: streamlit app that take image from user and detect the disease and print the image and the disease name

import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import glob

# Load the YOLOv8 model
model = YOLO(r"C:\Users\moham\Desktop\Real-Time-Object-Detection-with-YOLOv10-and-Webcam-Step-by-step-Tutorial-main\best.pt")

# Define class names (replace with actual class names)
class_names = ['Acne','Chickenpox','Eczema','Monkeypox','Pimple','Psoriasis','Ringworm','basal cell carcinoma','melanoma','tinea-versicolor','vitiligo','warts']  # Update with your class names

def process_image(image):
    """
    Process the uploaded image and return the detected disease and annotated image.
    """
    results = model.predict(image)
    result = results[0]
    annotated_frame = result.plot()  # Plot detections on the image
    detections = result.boxes.cls  # Get detected class indices
    
    # Get the class name with the highest confidence
    if detections.numel() > 0:
        predicted_class_index = int(detections[0].item())
        predicted_class = class_names[predicted_class_index]
    else:
        predicted_class = "No disease detected"
    
    return predicted_class, annotated_frame

# Streamlit app
st.title("Skin Disease Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Disease"):
        predicted_class, annotated_image = process_image(image)
        st.image(annotated_image, caption='Detected Disease', use_column_width=True)
        st.write("Detected Disease:", predicted_class)
        import streamlit as st

        # Dictionary of descriptions and recommendations
        disease_info = {
             "Acne": {
        "Description": "Common skin condition causing pimples.",
        "Products": "Benzoyl Peroxide, Salicylic Acid."
    },
    "Chickenpox": {
        "Description": "Viral infection causing itchy blisters.",
        "Recommendation": "See a doctor."
    },
    "Eczema": {
        "Description": "Chronic, itchy, inflamed skin.",
        "Products": "Eucerin, Aquaphor, Cortizone-10, Fucidin."
    },
    "Monkeypox": {
        "Description": "Viral infection causing rash and fever.",
        "Recommendation": "See a doctor."
    },
    "Pimple": {
        "Description": "Small pustule or papule.",
        "Products": "Benzoyl Peroxide, Salicylic Acid."
    },
    "Psoriasis": {
        "Description": "Scaly, itchy skin patches.",
        "Products": "Neutrogena T/Gel, CeraVe Psoriasis Cleanser, Aveeno."
    },
    "Ringworm": {
        "Description": "Fungal infection causing ring-shaped rash.",
        "Products": "Lotrimin, Lamisil, Tinactin."
    },
    "basal cell carcinoma": {
        "Description": "Slow-growing skin cancer.",
        "Recommendation": "See a doctor."
    },
    "melanoma": {
        "Description": "Dangerous skin cancer.",
        "Recommendation": "See a doctor."
    },
    "tinea-versicolor": {
        "Description": "Fungal infection causing discolored patches.",
        "Products": "Selsun Blue, Lamisil."
    },
    "vitiligo": {
        "Description": "Loss of skin color in patches.",
        "Recommendation": "See a doctor."
    },
    "warts": {
        "Description": "Skin growths caused by viruses.",
        "Products": "Compound W, Dr. Scholl's Freeze Off."
    }
        }

        if predicted_class in disease_info:
            st.write(f"Description: {disease_info[predicted_class]['Description']}")
            if 'Products' in disease_info[predicted_class]:
                st.write(f"Products: {disease_info[predicted_class]['Products']}")
            if 'Recommendation' in disease_info[predicted_class]:
                st.write(f"Recommendation: {disease_info[predicted_class]['Recommendation']}")


              