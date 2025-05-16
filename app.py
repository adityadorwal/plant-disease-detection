import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

st.title("Plant Disease Detection with Grad-CAM")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(num_ftrs, 38)
    )
    model.load_state_dict(torch.load("resnet18_plant_disease.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

def generate_gradcam(model, input_tensor, class_idx=None):
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    last_conv = model.layer4[-1].conv2
    forward_handle = last_conv.register_forward_hook(forward_hook)
    backward_handle = last_conv.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax().item()
    loss = output[0, class_idx]
    loss.backward()

    forward_handle.remove()
    backward_handle.remove()

    weights = gradients.mean(dim=[2, 3], keepdim=True)
    grad_cam_map = (weights * features).sum(dim=1, keepdim=True).relu()
    grad_cam_map = torch.nn.functional.interpolate(grad_cam_map, size=(224, 224), mode='bilinear', align_corners=False)
    grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
    grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)

    return grad_cam_map

def overlay_heatmap_pil(image, heatmap_array):
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_array)).convert("L")
    heatmap_img = ImageOps.colorize(heatmap_img, black="black", white="red")
    heatmap_img = heatmap_img.resize(image.size)
    blended = Image.blend(image, heatmap_img, alpha=0.5)
    return blended

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]

    st.success(f"Predicted Disease: **{prediction}**")

    if st.checkbox("Show Grad-CAM Heatmap"):
        heatmap = generate_gradcam(model, input_tensor, predicted.item())
        overlay = overlay_heatmap_pil(image.resize((224, 224)), heatmap)
        st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
