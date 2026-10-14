import streamlit as st
import cv2, numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Pothole Detector", page_icon="🛣️")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🛣️ Pothole Detector")
st.write("Upload a road photo. The AI outlines damaged areas and shows the damage %.")

file = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png"])
conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)

if file:
    image = np.array(Image.open(file).convert("RGB"))
    h, w = image.shape[:2]
    r = model.predict(image[:, :, ::-1], conf=conf, verbose=False)[0]

    col1, col2 = st.columns(2)
    col1.image(image, caption="Before", use_container_width=True)

    if r.masks is None:
        col2.image(image, caption="After", use_container_width=True)
        st.success("No damage found ✅")
    else:
        col2.image(r.plot()[:, :, ::-1], caption="After", use_container_width=True)
        total = 0
        for i, mk in enumerate(r.masks.data.cpu().numpy(), 1):
            pct = 100 * (cv2.resize(mk, (w, h)) > 0.5).sum() / (h * w)
            total += pct
            st.write(f"Pothole #{i}: {pct:.2f}% of the image")
        st.metric("Total damaged area", f"{total:.2f}%")
