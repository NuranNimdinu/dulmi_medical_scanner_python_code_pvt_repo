import time
import busio
import board
import adafruit_mlx90640
import cv2
import numpy as np
import os
from datetime import datetime

    # def data_ready(self) -> boolean:
    #     statusRegister = [0]
    #     self._I2CReadWords(0x8000, statusRegister)
    #     dataReady = statusRegister[0] & 0x0008
    #     # print("ready status: 0x%x" % dataReady)
    #     return 0 if dataReady == 0 else 1


class MLX90640_CAM_DRIVER:
    max_temp = 0.0
    min_temp = 0.0
    rgb_frame_size = (480,640)
    temp_raw_frame = np.zeros((24,32))
    temp_png_frame = None
    save_folder = "thermal_csv"
    png_save_folder = "thermal_png"
    filename_prefix = ""

    def __init__(self, dev_addr = 0x33, file_name_p = "T"):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c, dev_addr)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        self.filename_prefix = file_name_p

    def change_csv_save_path(self, png_path, csv_path):
        self.png_save_folder = png_path
        self.save_folder = csv_path
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(self.png_save_folder, exist_ok=True)

    def save_raw_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_folder, f"{self.filename_prefix}_thermal_raw_{timestamp}.csv")
        np.savetxt(filepath, self.temp_raw_frame, delimiter=",", fmt='%s')
        
        print(f"Successfully saved matrix to {filepath}")

    def save_temp_png(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.png_save_folder, f"{self.filename_prefix}_thermal_img_{timestamp}.png")
        cv2.imwrite(filepath, self.temp_png_frame)

    def get_image(self):
        try:
            if not self.mlx.data_ready():
                return None
            
            tframe = np.zeros((24*32,))

            try:
                self.mlx.getFrame(tframe)
            except Exception as e:
                return None
            
            tframe[290] = (tframe[290 - 1] + tframe[290 + 1]) / 2
            tframe = np.reshape(tframe, (24,32))
            tframe = np.rot90(tframe, k=1)

            self.temp_raw_frame = tframe

            self.max_temp = tframe.max()
            self.min_temp = tframe.min()
            rgb_frame = np.uint8((tframe - self.min_temp) * 255 / (self.max_temp - self.min_temp)) ## mapping to 0 - 255

            heat_image = self.resize_with_pad(rgb_frame, self.rgb_frame_size)
            # COLORMAP_INFERNO (black/purple/orange/yellow) and COLORMAP_JET (blue/green/red) are the best choices
            heat_image = cv2.applyColorMap(heat_image, cv2.COLORMAP_JET)

            return heat_image

        except Exception as e:
            print(f"error : {e}")
            return None
        
    def get_image_with_temp_grd(self):
        t_img = self.get_image();

        if t_img is None:
            return None
        
        text = 'Tmin = {:+.1f} Tmax = {:+.1f}'.format(self.get_min_temp(), self.get_max_temp())
        cv2.putText(t_img, text, (50, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        temp_bar_image = self.create_temperature_bar(self.rgb_frame_size[1], self.get_min_temp(), self.get_max_temp())

        spacer = np.ones((self.rgb_frame_size[1], 20, 3), dtype=np.uint8) * 255
        t_img = np.hstack((t_img, spacer, temp_bar_image))

        self.temp_png_frame = t_img
        return t_img
        
    def get_max_temp(self) -> float:
        return self.max_temp
    
    def get_min_temp(self) -> float:
        return self.min_temp
    
    def set_rgb_frame_size(self, w, h):
        self.rgb_frame_size = (w, h)

    def resize_with_pad(self, image_matrix, target_size=(640, 480), pad_color=(0, 0, 0)):
        """
        Resizes an image matrix to the target resolution without stretching.
        Maintains aspect ratio and pads the rest with a solid color.
        
        target_size: (width, height)
        pad_color: Tuple for BGR/RGB colors, or a single integer (e.g., 0) for grayscale
        """
        # 1. Extract current dimensions and target dimensions
        target_w, target_h = target_size
        h, w = image_matrix.shape[:2]

        # 2. Calculate the scaling factor to fit the image inside the target box
        scale = min(target_w / w, target_h / h)

        # 3. Calculate new dimensions based on the scale
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 4. Resize the image (this does not stretch, just shrinks/grows proportionally)
        resized_img = cv2.resize(image_matrix, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 5. Calculate the padding needed to center the image on the target canvas
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        # 6. Apply the padding border
        padded_matrix = cv2.copyMakeBorder(
            resized_img, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=pad_color
        )

        return padded_matrix

    def create_temperature_bar(self, height, min_temp, max_temp):
        bar_width = 50
        label_width = 100
        
        gradient = np.linspace(255, 0, height, dtype=np.uint8)
        gradient = np.repeat(gradient[:, np.newaxis], bar_width, axis=1)
        color_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        
        canvas = np.ones((height, bar_width + label_width, 3), dtype=np.uint8) * 255
        canvas[:, :bar_width] = color_bar
        
        num_ticks = 6
        for i in range(num_ticks):
            y = int(i * (height - 1) / (num_ticks - 1))
            temp = max_temp - (max_temp - min_temp) * (i / (num_ticks - 1))
            cv2.line(canvas, (bar_width, y), (bar_width + 5, y), (0, 0, 0), 1)
            text_y = y + 5
            if i == 0:
                text_y = y + 15
            elif i == num_ticks - 1:
                text_y = y - 5
                
            cv2.putText(canvas, f"{temp:.1f}", (bar_width + 15, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        
        return canvas
