# TEMPORARY CODE IN train_model.py
import os
print("Files in current directory:")
print(os.listdir(os.path.dirname(os.path.abspath(__file__))))

# Try a generic import of the module
import deblur_model
print("Module imported successfully!")

# Now try to access the function explicitly
try:
    model_fn = deblur_model.deblur_cnn_model
    print("Function successfully found and referenced!")
except AttributeError:
    print("!!! ERROR: Function name deblur_cnn_model was NOT found in the module.")

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import kagglehub
from deblur_model import deblur_cnn_model
from deblur_utils import load_data, calculate_psnr
import numpy as np
import matplotlib.pyplot as plt

# 1. Download the dataset and get the root path
DATASET_ROOT_PATH = kagglehub.dataset_download("jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets")

# 2. Define the list of dataset folders to use for training
# These names must match the folders inside the 'DBlur' directory.
DATASET_NAMES = ['CelebA', 'Gopro', 'Helen', 'Wider-Face'] 

# 3. Define the sub-folder structure common to most training sets
# We will look inside the 'train' folder of each dataset.
DATA_SPLIT_FOLDER = 'train' 
BLUR_SUBFOLDER = 'blur'
SHARP_SUBFOLDER = 'sharp'

# Base path to the DBlur directory
DBLUR_BASE_PATH = os.path.join(DATASET_ROOT_PATH, 'DBlur')

MODEL_DIR = 'model_weights'
MODEL_FILENAME = 'deblur_cnn.h5'
IMG_SIZE = 128
EPOCHS = 50 
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# --- Training Logic (MODIFIED FOR AGGREGATION) --- 
def train_deblur_model():
    print("✨ Starting Aggregated Data Loading...")
    
    X_all = [] # List to hold all blurry inputs
    Y_all = [] # List to hold all sharp targets

    # Loop through each dataset folder defined in the configuration
    for dataset_name in DATASET_NAMES:
        
        # Construct the full paths for the current dataset's training split
        current_blur_path = os.path.join(DBLUR_BASE_PATH, dataset_name, DATA_SPLIT_FOLDER, BLUR_SUBFOLDER)
        current_sharp_path = os.path.join(DBLUR_BASE_PATH, dataset_name, DATA_SPLIT_FOLDER, SHARP_SUBFOLDER)

        # Handle the HIDE, RealBlur_J, RealBlur_R datasets which sometimes lack a 'train' folder
        # If the standard path doesn't exist, try looking in a 'test' folder or directly inside
        if not os.path.exists(current_blur_path) or not os.path.exists(current_sharp_path):
            print(f"⚠️ Standard 'train' path not found for {dataset_name}. Checking alternative structure...")
            
            # Check for direct 'blur'/'sharp' folders inside the dataset_name directory (common in RealBlur/HIDE)
            current_blur_path = os.path.join(DBLUR_BASE_PATH, dataset_name, BLUR_SUBFOLDER)
            current_sharp_path = os.path.join(DBLUR_BASE_PATH, dataset_name, SHARP_SUBFOLDER)
            
            if not os.path.exists(current_blur_path) or not os.path.exists(current_sharp_path):
                print(f"🛑 Skipping {dataset_name}: Required data folders not found.")
                continue


        print(f"\n- Loading data from: {dataset_name}")
        X_data, Y_data = load_data(current_blur_path, current_sharp_path, target_size=(IMG_SIZE, IMG_SIZE))
        
        if len(X_data) > 0:
            X_all.append(X_data)
            Y_all.append(Y_data)
        else:
            print(f"No data found in {dataset_name}.")


    # Aggregate all loaded data into final NumPy arrays
    if not X_all:
        print("🛑 No data loaded from any source. Training aborted.")
        return

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)

    print(f"\n✅ Aggregated a total of {len(X)} image pairs for training.")
    
    # Split the aggregated data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=VALIDATION_SPLIT, random_state=42)

    # Define and Compile Model
    model = deblur_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Compile model using MSE loss (common for image reconstruction)
    model.compile(optimizer='adam', loss='mse')
    
    # Create Model Weights Directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    
    # Callbacks: Save the model only when validation loss improves
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=1)
    ]
    
    print("\n⏳ Starting model training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    print(f"🎉 Training complete. Best model saved to {model_save_path}")

    # Load best model for final evaluation
    best_model = tf.keras.models.load_model(model_save_path)
    
    # Evaluate on Validation set
    Y_pred = best_model.predict(X_val)
    
    # Calculate PSNR
    avg_psnr = np.mean([calculate_psnr(Y_val[i], Y_pred[i]) for i in range(len(Y_val))])
    print(f"\n📊 Average PSNR on Validation Set (Best Model): {avg_psnr:.4f} dB")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'training_loss.png'))
    # plt.show()

if __name__ == '__main__':
    train_deblur_model()