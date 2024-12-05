import streamlit as st
from PIL import Image
from ultralytics import YOLO
from src.custom_model.simple_model import SimpleCNN
import torch
import torchvision.transforms as transforms

@st.cache_resource

#For custom model this is all that needs to change
def load_model():
    model = SimpleCNN(num_classes=7)
    model.load_state_dict(torch.load("models/custom/custom_model.pt"))
    model.eval()  
    return model

model = load_model()
st.title("Simple CNN ") # App layout ( Change for custom model)
col1, col2 = st.columns(2)

with col1:
    st.header("Input Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
    else:
        input_image = None

#Col2 should just be the result of inference, so probably nothing changes here
with col2:
    st.header("Detection Result")
    if input_image:

        transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to a standard size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
        input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)

        st.write("You have Herpes")# Display the result
    else:
        st.write("Upload an image to see the detection results.")


