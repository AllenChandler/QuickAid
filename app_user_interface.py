import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.custom_model.simple_model import get_pretrained_model
import os
import openai
from transformers import BertTokenizer, BertForQuestionAnswering


# Set your OpenAI API key
openai.api_key = "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"

@st.cache_resource
def load_model():
    model = get_pretrained_model(num_classes=2)
    model.load_state_dict(torch.load("./models/custom/best_model.pt"))
    model.eval()  
    return model

# Load class names
class_names_path = os.path.abspath("./models/custom/class_names.txt")
with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f]

model = load_model()

# App layout
st.title("QuickAid")  # Updated title
col1, col2 = st.columns(2)

# Input Image
with col1:
    st.header("Input Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Uploaded Image", use_container_width=True)
    else:
        input_image = None

# Detection Result
with col2:
    st.header("Detection Result")
    if input_image:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(input_image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

        # Display the result
        st.write("\nClass Probabilities:")
        for cls, prob in zip(class_names, probabilities[0]):
            st.write(f"{cls}: {prob*100:.2f}%")

        if confidence < 0.9:
            st.write("\nPrediction: Uncertain")
            st.write("\nIf you believe you have a skin condition, try another photo. Common issues arise when the quality of the photo isn't clear, low illumination, or there are other things in the background.")
        else:
            result = class_names[predicted_class]
            st.write(f"\nPredicted Disease: {result} (Confidence: {confidence*100:.2f} %)")

    else:
        st.write("Upload an image to see the detection results.")




model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
st.subheader("Chat with AI about your condition")
user_input = st.text_input("Ask the chatbot about your condition:")

if st.button("Ask"):
    if user_input.strip() != "":
        # Context about eczema
        if confidence >= .9:
            if result == 'eczema':
                context = """
                Eczema, also known as atopic dermatitis, is a condition that makes the skin red, inflamed, and itchy. 
                It is common in children but can occur at any age. Eczema is a chronic condition that flares up periodically. 
                Common symptoms include itching, red or inflamed skin, dryness, and rashes. It can be triggered by allergens, stress, or irritants.
                """
            # context about chickenpox
            if result == 'chickenpox':
                context = """
                Chickenpox, also known as varicella, is a highly contagious disease caused by the varicella-zoster virus. 
                It is characterized by an itchy skin rash with red spots and fluid-filled blisters, which eventually scab over. 
                Chickenpox is most common in children but can also affect adults. The rash usually begins on the chest, back, or face and then spreads to the rest of the body. 
                Other symptoms include fever, fatigue, and loss of appetite. The disease is highly contagious and spreads through direct contact with the rash or through respiratory droplets. 
                Chickenpox is generally mild in children but can lead to complications in adults or those with weakened immune systems. Vaccination is available and is the most effective way to prevent the disease.
                """

        
            inputs = tokenizer.encode_plus(user_input, context, add_special_tokens=True, return_tensors='pt')
            outputs = model(**inputs)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)
            answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            st.write(f"Chatbot: {answer}")
        
        else:
            st.warning("Cannot assist you at this time")
    else:
        st.warning("Please enter a question to ask the chatbot.")