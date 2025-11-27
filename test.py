# streamlit_gradio_api_app.py

import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
import tempfile

# ---------------------
# Connect to your API
# ---------------------
client = Client("Mohamed192003/image-classification-detection-system")

# ---------------------
# Streamlit UI
# ---------------------
st.title("Image Classification App (via Gradio API)")
st.write("Upload an image and the HF Space will return the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save to temp file so gradio_client can handle it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        temp_path = tmp.name

    st.write("Classifying... Please wait.")

    try:
        result = client.predict(
            image=handle_file(temp_path),
            api_name="/predict"
        )

        st.subheader("Prediction Result:")
        st.write(result)

    except Exception as e:
        st.error(f"Error calling the model API: {e}")

