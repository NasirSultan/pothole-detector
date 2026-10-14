import streamlit as st
import cv2, numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Pothole Detector", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def draw_segments(image, masks, alpha=0.5):
    """Fill each pothole with color and draw a clear outline."""
    colors = [(255, 0, 0), (0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255)]
    overlay = image.copy()
    for i, m in enumerate(masks):
        overlay[m == 1] = colors[i % len(colors)]
    out = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    for i, m in enumerate(masks):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colors[i % len(colors)], 3)
    return out

st.title("Pothole Detector")
st.write("Upload a road photo. The AI fills damaged areas with color and shows the damage percentage.")

file = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png"])
c1, c2 = st.columns(2)
conf = c1.slider("Confidence (lower = finds more)", 0.05, 0.9, 0.25, 0.05)
alpha = c2.slider("Fill strength", 0.1, 0.9, 0.5, 0.1)

if file:
    image = np.array(Image.open(file).convert("RGB"))
    h, w = image.shape[:2]
    r = model.predict(image[:, :, ::-1], conf=conf, verbose=False)[0]

    col1, col2 = st.columns(2)
    col1.image(image, caption="Before", use_container_width=True)

    if r.masks is None:
        col2.image(image, caption="After", use_container_width=True)
        st.success("No damage found. Try a lower confidence.")
    else:
        masks = [(cv2.resize(m, (w, h)) > 0.5).astype(np.uint8)
                 for m in r.masks.data.cpu().numpy()]
        col2.image(draw_segments(image, masks, alpha),
                   caption="After (filled segments)", use_container_width=True)

        total = 0
        for i, m in enumerate(masks, 1):
            pct = 100 * m.sum() / (h * w)
            total += pct
            st.write(f"Pothole #{i}: {pct:.2f}% of the image")
        st.metric("Total damaged area", f"{total:.2f}%")