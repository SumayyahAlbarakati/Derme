# prompt: streamlit app that take image from user and detect the disease and print the image and the disease name

import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import glob

# Load the YOLOv8 model
model = YOLO(r"best.pt")

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
def vertical_line(color="black"):
    """Creates an HTML element styled as a vertical line with a customizable color.

    Args:
        color (str, optional): The color of the vertical line. Defaults to "black".
    """
    html_temp = f"""
    <div style="width: 2px; height: 100%; background-color: {color}; margin: 0 auto;"></div>
    """
    st.components.v1.html(html_temp)

 


st.set_page_config(page_title="Derme",layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Create two columns
column1, column2, column3 = st.columns([0.45,0.1,0.45])

# Add content to each column
with column1:
    st.title("Skin Disease Detection App")
    logo= Image.open(r"Images/DermeLogo.png")
    st.image(logo, caption='Uploaded Image', width=150)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
with column2:
    pass
    

if uploaded_file is not None:
    with column1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown("""
            <style>
            .wide-button {
              width: 80%;  /* Adjust percentage for desired width (less than 100% for padding) */
              margin: 0 auto;  /* Center the button horizontally */
              padding: 10px 20px;  /* Add some padding for better visual appearance */
            }
            </style>
        """, unsafe_allow_html=True)
        
        detect_button = st.button("Detect Disease")
    with column3:
        if detect_button:
            st.header("Result")
            predicted_class, annotated_image = process_image(image)
            st.image(annotated_image, caption='Detected Disease', use_column_width=True)
            st.write("Detected Disease:", predicted_class)
            import streamlit as st

            # Dictionary of descriptions and recommendations
            disease_info = {
            "Acne": {
                "Description": "a common skin condition that happens when hair follicles under the skin become clogged.",
                "Source":r"https://www.niams.nih.gov/health-topics/acne",
                "Products": ["Benzoyl Peroxide", "Salicylic Acid"],
                "Images": [Image.open(r"Images/BenzoylPeroxide.jpg"), Image.open(r"Images/SalicylicAcid.jpg")]
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
                "Description": "It is a common fungal infection caused by dermatophytes (microscopic organisms that live on the dead, outer layer of the skin). The fungi that cause the rash appear as a ring and a raised, scaly edge on the skin.",
                "Source":r"https://www.moh.gov.sa/en/HealthAwareness/EducationalContent/Diseases/Dermatology/Pages/005.aspx",
                "Products": ["Lotrimin", "Lamisil", "Tinactin"],
                "Images": [Image.open(r"Images/Lotrimin.jpg"), Image.open(r"Images/Lamisil.jpg"),Image.open(r"Images/Tinactin.jpg")]
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

            # Assuming you have a variable `predicted_class` containing the disease prediction
            if predicted_class in disease_info:
              disease_data = disease_info[predicted_class]  # Access specific disease information

              # Display title
              st.header(f"{predicted_class} Information")
              description_with_source = f"{disease_data['Description']} [Source]({disease_data['Source']})"

              # Display description
              st.markdown(disease_data["Description"])

              # Display source with link
              st.markdown(f"Source: [Ministry of Health, Saudi Arabia]({disease_data['Source']})")

              # Display product list with a bullet list
              if 'Products' in disease_data:
                st.subheader("Products")
                product_list = st.empty()  # Create an empty container for bullet points
                for product in disease_data["Products"]:
                  product_list.markdown(f"- {product}")

              # Display images in a row using columns
              if 'Images' in disease_data:
                st.subheader("Images")
                col1, col2, col3 = st.columns(3)  # Create three columns for image layout
                for i, image in enumerate(disease_data["Images"]):
                  if i < 3:  # Display only the first 3 images (adjust as needed)
                    with col1 if i == 0 else col2 if i == 1 else col3:
                      st.image(image)
                  else:
                    # Handle potential overflow of images (optional)
                    st.warning(f"There are more than 3 images. Only displaying the first 3.")
                    break

              # Recommendation (if available)
              if 'Recommendation' in disease_data:
                st.subheader("Recommendation")
                st.write(disease_data["Recommendation"])



              
