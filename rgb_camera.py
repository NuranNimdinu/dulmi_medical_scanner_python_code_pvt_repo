import cv2 as cv
import numpy as np
import os
from datetime import datetime

class CAMERA_DRIVER:
    USE_PICAMERA = True
    USB_CAMERA_ID = 0
    csv_save_folder = "intensity_matrices"
    png_save_folder = "rgb_images"
    temp_image = None

    ### Initialize the camera driver with specified resolution and camera type
    ### resolution: tuple (width, height) for the camera resolution
    ### raspCam: boolean, True for Raspberry Pi Camera, False for USB Camera
    ### cameraID: integer, ID for the USB camera (any number) and pi cam (0 or 1) (default is 0)
    def __init__(self, resolution=(640,480), raspCam=True, cameraID=0):
        self.USE_PICAMERA = raspCam
        self.USB_CAMERA_ID = cameraID

        if self.USE_PICAMERA:
            from picamera2 import Picamera2
            self.picam2 = Picamera2(camera=cameraID)
            picamconfig = self.picam2.create_video_configuration(main={"size": resolution})
            self.picam2.configure(picamconfig)
            self.picam2.start()

        else:
            self.cam_cap = cv.VideoCapture(self.USB_CAMERA_ID, cv.CAP_V4L2)
            self.cam_cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
            self.cam_cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cam_cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cam_cap.set(cv.CAP_PROP_FPS, 30)
            
            if not self.cam_cap.isOpened():
                print("Error: Cannot open camera")
            _ = self.cam_cap.read()

    def capture(self):
        if self.USE_PICAMERA:
            rgb = self.picam2.capture_array()
            frame = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            self.temp_image = frame
            return True, frame
        else:
            self.temp_image = self.cam_cap.read()
            return self.temp_image
        
    def change_image_save_path(self, png_path, csv_path):
        self.csv_save_folder = csv_path
        self.png_save_folder = png_path
        os.makedirs(self.png_save_folder, exist_ok=True)
        os.makedirs(self.csv_save_folder, exist_ok=True)
        
    def save_image_png(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.png_save_folder, f"{timestamp}_rgb.png")
        cv.imwrite(filepath, self.temp_image)

    def save_intensity_matrix(self):
        
        # Convert standard OpenCV BGR image to a 2D Intensity (Grayscale) Matrix
        # This reduces the 3D array (Height x Width x 3) to a 2D array (Height x Width)
        # using the standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
        intensity_matrix = cv.cvtColor(self.temp_image, cv.COLOR_BGR2GRAY)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.csv_save_folder, f"{timestamp}_intensity.csv")
        np.savetxt(filepath, intensity_matrix, delimiter=",", fmt='%d')
        
        print(f"Saved {intensity_matrix.shape} intensity matrix to {filepath}")

    def close(self):
        if self.USE_PICAMERA:
            self.picam2.stop()
        else:
            self.cam_cap.release()