import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time
from deblur_utils import preprocess_image_for_model, postprocess_image

# --- Configuration ---
MODEL_PATH = 'model_weights/deblur_cnn.h5'
MODEL_INPUT_SIZE = (128, 128)

@st.cache_resource
def load_deblur_model():
    """Load the trained Keras model, cached for fast Streamlit runs."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}. Please train the model first by running train_model.py.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def deblur_image(model, image):
    """Processes image through the model and returns the deblurred image."""
    # Preprocess (resize, normalize, add batch dimension)
    input_tensor = preprocess_image_for_model(image, target_size=MODEL_INPUT_SIZE)
    
    # Predict
    output_tensor = model.predict(input_tensor, verbose=0)
    
    # Postprocess (remove batch dimension, denormalize, convert to 8-bit)
    deblurred_array = postprocess_image(output_tensor)
    
    return Image.fromarray(deblurred_array)

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="AI Image Deblur Tool", layout="wide")
    st.title("📸 AI-Powered Image Deblurring")
    st.markdown("Upload a blurry image, and our CNN Autoencoder will attempt to restore it.")

    # Load Model
    model = load_deblur_model()
    if model is None:
        st.warning("Model loading failed. Please ensure you have successfully run `train_model.py`.")
        return

    # File Uploader
    uploaded_file = st.file_uploader("Choose a blurry image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the original image
        original_image = Image.open(uploaded_file).convert("RGB")
        
        st.subheader("Results")
        
        # Create two columns for comparison
        col1, col2 = st.columns(2)
        
        # Deblur and time the process
        with st.spinner("⏳ Deblurring in progress..."):
            try:
                start_time = time.time()
                deblurred_image = deblur_image(model, original_image)
                end_time = time.time()
                inference_speed = (end_time - start_time) * 1000 # in ms
            except Exception as e:
                st.error(f"An error occurred during deblurring: {e}")
                return

        # Display Images
        with col1:
            st.image(original_image, caption=f'Uploaded Blurry Image (Original Size: {original_image.size[0]}x{original_image.size[1]})', use_column_width=True)
        with col2:
            # Resize the output image back to the original size for better comparison display
            resized_output = deblurred_image.resize(original_image.size)
            st.image(resized_output, caption='Deblurred Image (CNN Output)', use_column_width=True)
            
        st.markdown("---")
        st.info(f"**Inference Speed (on 128x128 input):** {inference_speed:.2f} ms")

        # Download Button
        from io import BytesIO
        buf = BytesIO()
        resized_output.save(buf, format="PNG")
        st.download_button(
            label="Download Deblurred Image",
            data=buf.getvalue(),
            file_name="deblurred_output.png",
            mime="image/png"
        )
        
    else:
        st.info("⬆️ Upload an image to start the deblurring process.")

if __name__ == '__main__':
    main()