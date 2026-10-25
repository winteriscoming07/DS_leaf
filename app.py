"""Upload a leaf image -> predicted species, confidence, accepted/rejected. Run: streamlit run app.py"""
import streamlit as st
from PIL import Image

from src.predict import LeafPredictor

st.title("Can your AI admit it doesn't know?")
st.caption("MobileNetV3-Small trained on 5 species (Apple, Corn, Grape, Peach, Tomato) from PlantVillage.")


@st.cache_resource
def get_predictor():
    return LeafPredictor("mobilenet")


pr = get_predictor()
f = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
if f:
    img = Image.open(f).convert("RGB")
    st.image(img, width=300)
    r = pr.predict_images([img])[0]
    st.metric("Best guess", r["predicted"], f"confidence {r['confidence']:.3f} (threshold {r['threshold']:.3f})", delta_color="off")
    (st.success if r["accepted"] else st.warning)(f"Outcome: {r['outcome']}")
    st.info("Rejection is a confidence-based safeguard, not proof that the plant is botanically unknown. "
            "Acceptance is not proof of correctness: unfamiliar leaves can be accepted with high confidence.")
