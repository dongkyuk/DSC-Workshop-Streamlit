import torch
import streamlit as st
import PIL.Image as Image
import numpy as np
from model import ConvModelClass

# st.title("A Simple Streamlit Web App")
st.title("Workshop Traffic Sign Classifier")
st.write("This is a simple web app to classify traffic signs.")

# Load pytorch model
model = ConvModelClass()
model.to("cpu")
model.load_state_dict(torch.load("checkpoint.pth", map_location=torch.device('cpu'))['model_state_dict'])

classes = ['Traffic Light', 'Speed Limit', 'Crosswalk', 'Stop Sign']

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

# Show image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image.putalpha(255)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    image = image.resize((256, 256))
    image = np.array(image).reshape(1, 4, 256, 256)
    prediction = torch.argmax(model(torch.from_numpy(image).float()))
    st.write(f"**Result:** {classes[prediction]}")

take_a_pic = st.checkbox("Take a picture")
if take_a_pic:
    picture = st.camera_input("Take a picture")
    if picture is not None:
        st.write("Classifying...")
        image = Image.open(picture).convert('RGB')
        image.putalpha(255)
        image = image.resize((256, 256))
        image = np.array(image).reshape(1, 4, 256, 256)
        prediction = torch.argmax(model(torch.from_numpy(image).float()))
        st.write(f"**Result:** {classes[prediction]}")
