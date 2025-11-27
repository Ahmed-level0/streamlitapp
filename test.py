import streamlit as st
from gradio_client import Client
from PIL import Image
import tempfile

st.title("Gradio API Image Classifier")

client = Client("Mohamed192003/image-classification-detection-system")

uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        temp_path = tmp.name

    # Call Gradio API
    result = client.predict(
        image=temp_path,
        api_name="/predict"
    )

    st.subheader("Prediction Result")
    st.write(result)
