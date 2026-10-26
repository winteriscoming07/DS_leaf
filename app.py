"""Upload a leaf image -> predicted species, confidence, accepted/rejected. Run: streamlit run app.py"""
import streamlit as st

from src.predict import LeafPredictor, open_upload

st.title("Can your AI admit it doesn't know?")
st.caption("MobileNetV3-Small trained on 5 species (Apple, Corn, Grape, Peach, Tomato) from PlantVillage lab-style photographs. "
           "Potato and Bell pepper were never seen in training.")


@st.cache_resource
def get_predictor():
    return LeafPredictor("mobilenet")


pr = get_predictor()
f = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
if f:
    try:
        img = open_upload(f)
    except ValueError as e:
        st.error(str(e))
    else:
        st.image(img, width=300)
        r = pr.predict_images([img])[0]
        st.metric("Best guess", r["predicted"], f"confidence {r['confidence']:.3f} (threshold {r['threshold']:.3f})", delta_color="off")
        (st.success if r["accepted"] else st.warning)(f"Outcome: {r['outcome']}")
st.info("**Unknown / Uncertain is a confidence-based rejection, not a botanical identification.** It means the model's top probability "
        "fell below a threshold fixed on validation images of the five known species. Acceptance is not proof of correctness: "
        "unfamiliar leaves can be accepted with high confidence (e.g. Potato called Tomato), and softmax confidence is uncalibrated. "
        "Results come from controlled PlantVillage images; field photographs have not been tested.")
