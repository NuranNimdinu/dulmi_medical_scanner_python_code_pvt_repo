import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

class RGBUlcerPredictor:
    def __init__(self, model_path="/home/pi/data/dfu_resnet50_model.keras", threshold = 0.5):
        self.detection_threshold = threshold

        # Load the model ONCE when the class is initialized
        print(f"Loading RGB model from {model_path}...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_matrix(self, rgb_matrix):
        """Processes a raw 3D numpy array (RGB) into the tensor expected by ResNet50."""
        # 1. Resize the image to 224x224 as required by ResNet-50
        img_resized = cv2.resize(rgb_matrix, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # 2. Convert to a float32 array
        img_array = img_resized.astype(np.float32)
        
        # 3. Add batch dimension (turns 224,224,3 into 1,224,224,3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 4. Apply ResNet's specific preprocessing (converts RGB to BGR and zero-centers)
        processed_input = preprocess_input(img_array)
        
        return processed_input

    def predict(self, rgb_matrix, save_image=False, save_path="/home/pi/outputs", prefix="run"):
        """
        Runs inference on a numpy RGB matrix. 
        Returns: label (str), confidence (float), raw_probability (float)
        """
        if not self.model:
            print("[Error] Cannot predict. Model is not loaded.")
            return None, 0.0, 0.0

        # Preprocess the numpy matrix directly
        processed_input = self.preprocess_matrix(rgb_matrix)

        # Predict probability
        prob = float(self.model.predict(processed_input, verbose=0)[0][0])
        
        # Note: Original code logic uses <= for DM/Ulcer detection
        is_dm_ulcer = prob <= self.detection_threshold

        label = "DM / High Risk (Ulcer)" if is_dm_ulcer else "Control / Healthy"
        
        # Fixed syntax error: 1.0 - prob
        confidence = (1.0 - prob) if is_dm_ulcer else prob

        # Trigger the image saving function if requested
        if save_image:
            self._save_visualization(rgb_matrix, label, confidence, is_dm_ulcer, save_path, prefix)

        return label, confidence, prob

    def _save_visualization(self, raw_matrix, label, confidence, is_dm_ulcer, save_path, prefix):
        """Internal function to save the matplotlib plot as a PNG without displaying it"""
        # Ensure the destination folder exists
        os.makedirs(save_path, exist_ok=True)
        
        # Generate the exact filename requested
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_rgb_model_output_{timestamp}.png"
        filepath = os.path.join(save_path, filename)

        # Create the plot
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(raw_matrix)
        plt.axis('off') # Hide the axis ticks/numbers

        # Title the figure with the prediction
        color = 'red' if is_dm_ulcer else 'green'
        plt.title(f"AI Prediction: {label}\nConfidence: {confidence:.1%}", 
                  fontsize=14, fontweight='bold', color=color)
        plt.tight_layout()

        # Save the figure to the SD card
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        
        # CRITICAL: Close the figure to free up RAM!
        plt.close(fig) 
        
        print(f"Saved visualization: {filepath}")