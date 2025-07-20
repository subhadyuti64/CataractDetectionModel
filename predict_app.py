import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Cataract Detection", layout="centered")


class_names = ["immature", "mature"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("/Users/subhadyutirath/Documents/CataractDetectionModel/resnet18_cataract_model.pkl"))
    model.eval()
    return model

model = load_model()

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        return class_names[pred.item()]

st.title("👁️ Cataract Detection App")
st.markdown("Upload a retinal image to detect if the cataract is **Mature** or **Immature**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            prediction = predict_image(image)
            st.success(f"🧠 **Prediction:** {prediction.upper()}")
