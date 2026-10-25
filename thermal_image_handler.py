import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime

class ThermalUlcerPredictor:
    def __init__(self, model_path="/home/pi/data/best_thermal_model.keras", threshold=0.7, min_temp = 20.0, max_temp = 40.0):
        # --- NORMALIZATION VARIABLES ---
        self.MIN_TEMP = min_temp
        self.MAX_TEMP = max_temp

        self.detection_threshold = threshold
        
        # Load the model ONCE when the class is initialized
        print(f"Loading model from {model_path}...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_matrix(self, temp_matrix):
        """Processes a raw 2D numpy array into the 3D tensor expected by the model."""
        # Ensure data is float32
        temp_matrix = temp_matrix.astype(np.float32)

        foot_mask = (temp_matrix > 0).astype(np.uint8)

        mask_resized = cv2.resize(foot_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        temp_resized = cv2.resize(temp_matrix, (224, 224), interpolation=cv2.INTER_LINEAR)

        normalized = (temp_resized - self.MIN_TEMP) / (self.MAX_TEMP - self.MIN_TEMP)
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = 0.1 + (normalized * 0.9)

        final_matrix = np.where(mask_resized == 1, normalized, 0.0)

        # Fixed syntax error: Multiply the tuple by 3 to stack 3 channels
        tensor = np.stack((final_matrix,) * 3, axis=-1).astype(np.float32)
        
        return np.expand_dims(tensor, axis=0)

    def predict(self, temp_matrix, save_image=False, save_path="/home/pi/outputs", prefix="run"):
        """
        Runs inference on a numpy matrix. 
        Returns: label (str), confidence (float), raw_probability (float)
        """
        if not self.model:
            print("[Error] Cannot predict. Model is not loaded.")
            return None, 0.0, 0.0

        # Preprocess the numpy matrix directly
        processed_input = self.preprocess_matrix(temp_matrix)

        # Predict probability
        prob = float(self.model.predict(processed_input, verbose=0)[0][0])
        is_dm_ulcer = prob >= self.detection_threshold

        label = "DM / High Risk (Ulcer)" if is_dm_ulcer else "Control / Healthy"
        confidence = prob if is_dm_ulcer else (1.0 - prob)

        # Trigger the image saving function if requested
        if save_image:
            self._save_visualization(temp_matrix, processed_input, label, confidence, is_dm_ulcer, save_path, prefix)

        return label, confidence, prob

    def _save_visualization(self, raw_matrix, processed_input, label, confidence, is_dm_ulcer, save_path, prefix):
        """Internal function to save the matplotlib plot as a PNG without displaying it"""
        # Ensure the destination folder exists
        os.makedirs(save_path, exist_ok=True)
        
        # Generate the exact filename requested
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_thermal_model_output_{timestamp}.png"
        filepath = os.path.join(save_path, filename)

        # Create the plot
        model_view = processed_input[0, :, :, 0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(raw_matrix, cmap='inferno', vmin=self.MIN_TEMP, vmax=self.MAX_TEMP)
        axes[0].set_title("Raw Temperature Data")
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label=f"Temperature ({self.MIN_TEMP}°C to {self.MAX_TEMP}°C)")

        im2 = axes[1].imshow(model_view, cmap='inferno', vmin=0.0, vmax=1.0)
        axes[1].set_title("Processed Model Input")
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Normalized AI Scale [0.0 - 1.0]")

        color = 'red' if is_dm_ulcer else 'green'
        plt.suptitle(f"AI Prediction: {label} ({confidence:.1%})", fontsize=14, fontweight='bold', color=color)
        plt.tight_layout()

        # Save the figure to the SD card
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        
        # CRITICAL: Close the figure to free up RAM!
        plt.close(fig) 
        
        print(f"Saved visualization: {filepath}")