import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- Image Loading and Processing ---

def load_data(blurry_path, sharp_path, target_size=(128, 128)):
    """
    Loads and preprocesses image pairs by aligning filenames between blurry and sharp directories.
    """
    print(f"Loading data from: {blurry_path} and {sharp_path}")
    
    if not os.path.exists(blurry_path) or not os.path.exists(sharp_path):
        print("🛑 Data paths do not exist. Check your DATASET_ROOT_PATH and subdirectories.")
        return np.array([]), np.array([])
        
    blurry_files_list = os.listdir(blurry_path)
    
    X = [] # Blurry inputs
    Y = [] # Sharp targets
    
    count = 0

    for filename in blurry_files_list:
        blurry_file_path = os.path.join(blurry_path, filename)
        sharp_file_path = os.path.join(sharp_path, filename) 

        # 1. Ensure the pair exists (assuming filenames are identical)
        if not os.path.exists(sharp_file_path):
            continue 
            
        # Filter for common image extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            # 2. Load images (cv2 loads BGR by default)
            blurry_img = cv2.imread(blurry_file_path)
            sharp_img = cv2.imread(sharp_file_path)
            
            if blurry_img is None or sharp_img is None:
                continue
            
            # 3. Resize (to ensure consistent input shape)
            blurry_img = cv2.resize(blurry_img, target_size)
            sharp_img = cv2.resize(sharp_img, target_size)

            # 4. Convert to RGB and Normalize to [0, 1]
            blurry_img = cv2.cvtColor(blurry_img, cv2.COLOR_BGR2RGB)
            sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
            
            X.append(blurry_img / 255.0)
            Y.append(sharp_img / 255.0)
            count += 1
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    X = np.array(X, dtype="float32")
    Y = np.array(Y, dtype="float32") 
    
    print(f"Successfully loaded {count} image pairs.")
    return X, Y

# --- Evaluation Metric ---

def calculate_psnr(original_image, reconstructed_image):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in dB.
    Inputs should be numpy arrays with pixel values between 0.0 and 1.0.
    """
    # Rescale to [0, 255] integer range for PSNR calculation
    original = np.clip(original_image * 255.0, 0, 255).astype(np.uint8)
    reconstructed = np.clip(reconstructed_image * 255.0, 0, 255).astype(np.uint8)
    
    return psnr(original, reconstructed, data_range=255)

# --- Preprocessing for Inference (Streamlit app) ---

def preprocess_image_for_model(image_array, target_size=(128, 128)):
    """Resizes, normalizes, and adds batch dimension for model inference."""
    # Assuming PIL image input from Streamlit
    img = np.array(image_array.resize(target_size))
        
    img = img.astype("float32") / 255.0
    # Add batch dimension (1, H, W, C)
    return np.expand_dims(img, axis=0)

def postprocess_image(model_output):
    """Removes batch dimension, denormalizes, and converts to 8-bit integer."""
    # Remove batch dimension
    img = np.squeeze(model_output, axis=0)
    # Denormalize and clip to [0, 255]
    img = np.clip(img * 255.0, 0, 255).astype('uint8')
    return img