import asyncio
import base64
import glob
import json
import time
from tracemalloc import start
import websockets
import socket
import datetime as dt
import queue
import cv2
import numpy as np
from datetime import datetime
import threading
import os
import re
import sys
import subprocess
from gpiozero import DigitalOutputDevice

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from mlx90640_cam_drv import MLX90640_CAM_DRIVER
from rgb_camera import CAMERA_DRIVER
from web_app_files.web_app_launcher import run_web_app, get_server_status
from thermal_image_handler import ThermalUlcerPredictor
from rgb_image_handler import RGBUlcerPredictor
from ws2812_led_indicator import WS2812BThread
from push_button_hanlder import ButtonMonitor

### CONFIG ###
global_settings = {
    "T" : "SETTINGS",
    "SCAN_SAVE_IMGS_NUM" : 10,
    "SCAN_TIME_INTERVAL_SEC" : 1,
    "MIN_TEMP" : 20.0,
    "MAX_TEMP" : 40.0,
    "THERMAL_MODEL_THREAHOLD" : 0.5,
    "RGB_MODEL_THREAHOLD" : 0.5,
    "THERMAL_MODEL_PATH" : "/home/rasp3/project/medical_scanner/image_models/best_thermal_model.keras",
    "RGB_MODEL_PATH" : "/home/rasp3/project/medical_scanner/image_models/dfu_resnet50_model.keras",
    "save_images_base_path" : "medical_scanner_data/",
}

thermal_camera_L = MLX90640_CAM_DRIVER(0x32, "L")
thermal_camera_R = MLX90640_CAM_DRIVER(0x33, "R")
rgb_cameraL = CAMERA_DRIVER((1280, 720), rotation=-90, raspCam=True, cameraID=0)
rgb_cameraR = CAMERA_DRIVER((1280, 720), rotation=-90, raspCam=True, cameraID=1)


class SERVER_DATA_STRUCT:
    ws_client_connected = False
    ws_camera_stream_on = False
    ws_client_handler:websockets.ServerConnection
    ws_camera_stream_task:asyncio.Task
    wss_camera_feed_q = queue.Queue(maxsize=10)
    wss_data_tx_q = queue.Queue(maxsize=10)

server_data = SERVER_DATA_STRUCT()


### useful functions
def create_unique_folder(base_path, prefix="data"):
    """
    Creates a new folder with an auto-incrementing number (e.g., data_000001).
    Survives script restarts by scanning existing folders.
    """
    # 1. Ensure the main base directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # 2. Setup a safe pattern to look for existing folders matching our prefix
    # This matches exactly "prefix_number" (e.g., "data_000001")
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    
    highest_num = 0
    
    # 3. Scan the directory to find the highest existing number
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                # Extract the number and update highest_num if it's larger
                num = int(match.group(1))
                if num > highest_num:
                    highest_num = num
                    
    # 4. Calculate the next number and format it with 6 leading zeros
    next_num = highest_num + 1
    new_folder_name = f"{prefix}_{next_num:06d}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    
    # 5. Create and return the new unique folder
    os.makedirs(new_folder_path)
    return new_folder_path

class JsonConfigManager:
    def __init__(self, config_dict, file_path):
        """
        Initializes the configuration manager.
        
        Args:
            config_dict (dict): The global dictionary (defaults). Modified IN-PLACE.
            file_path (str): The path to the JSON file.
        """
        self.data = config_dict  # Direct reference (no deepcopy)
        self.file_path = file_path

    def save(self):
        """Saves the linked dictionary to the JSON file."""
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, indent=4)

    def load_and_sync(self):
        """
        Loads the JSON file and modifies the linked dictionary in-place.
        - Overwrites defaults with saved values.
        - Keeps default values for missing keys.
        - Ignores outdated/extra keys from the file.
        """
        if not os.path.exists(self.file_path):
            print(f"[{self.file_path}] Not found. Creating new file with default values.")
            self.save()
            return

        with open(self.file_path, 'r', encoding='utf-8') as file:
            try:
                loaded_dict = json.load(file)
            except json.JSONDecodeError:
                loaded_dict = {}

        loaded_keys = set(loaded_dict.keys())
        current_keys = set(self.data.keys())
        needs_saving = loaded_keys != current_keys

        # Modify the global dictionary in-place
        for key in current_keys:
            if key in loaded_dict:
                # File has the key: overwrite the default with the saved value
                self.data[key] = loaded_dict[key]
            else:
                # File is missing the key: the global dict already has the default value
                print(f"Added new missing key to {self.file_path}: '{key}'")

        removed_keys = loaded_keys - current_keys
        if removed_keys:
            print(f"Ignored outdated keys from {self.file_path}: {', '.join(removed_keys)}")

        # Clean up the file if it had extra/missing keys
        if needs_saving:
            self.save()
            print(f"[{self.file_path}] Keys synced and file updated.")

def check_for_updates_and_restart():
    print("Checking GitHub for updates...")
    try:
        # 1. Fetch the latest info from the remote repository
        subprocess.run(["git", "fetch", "--all"], check=True, capture_output=True)
        
        # 2. Check if the local branch is behind the remote branch
        status = subprocess.run(["git", "status", "-uno"], capture_output=True, text=True)
        
        if "Your branch is behind" in status.stdout:
            print("Update found! Downloading latest code...")
            
            # 3. Force update to exactly match the remote (prevents merge conflicts)
            # CHANGE 'main' TO 'master' if your repo uses the older naming!
            subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
            
            print("Update complete. Restarting program to apply changes...")
            
            # 4. Instantly restart the Python script so the new code runs
            try:
                os.closerange(3, 65535) # Close ports just like your restart function
            except:
                pass
            os.execv(sys.executable, [sys.executable] + sys.argv)
            
        else:
            print("App is fully up to date. Starting normally...")
            
    except Exception as e:
        print(f"Failed to check for updates: {e}")
        print("Continuing with the current version...")
#####################

### image hanlder thread
class ImageProcessWorker(threading.Thread):
    current_scan_folder_path = ""
    current_scan_num_folder_path = ""
    current_scan_count = 0
    current_scan_max_thermal_conf_L = 0.0
    current_scan_max_thermal_conf_R = 0.0
    current_scan_max_rgb_conf_L = 0.0
    current_scan_max_rgb_conf_R = 0.0

    is_scanning = False

    def __init__(self, server_st: SERVER_DATA_STRUCT):
        super().__init__(daemon=True)
        self._stop_worker = threading.Event()
        self._ws_server = server_st

        self.thermal_predictor = ThermalUlcerPredictor(global_settings["THERMAL_MODEL_PATH"], global_settings["THERMAL_MODEL_THREAHOLD"], global_settings["MIN_TEMP"], global_settings["MAX_TEMP"])
        self.rgb_predictor = RGBUlcerPredictor(global_settings["RGB_MODEL_PATH"], global_settings["RGB_MODEL_THREAHOLD"])

    def stop(self):
        self._stop_worker.set()

    def run(self):
        print("Image Processor worker Started!")

        while not self._stop_worker.is_set():
            ## capture images
            th_img_L = thermal_camera_L.get_image_with_temp_grd()
            th_img_R = thermal_camera_R.get_image_with_temp_grd()

            _, rgb_img_L = rgb_cameraL.capture()
            _, rgb_img_R = rgb_cameraR.capture()

            img_arr = [th_img_L, th_img_R, rgb_img_L, rgb_img_R]
            img_head_tag = ["tl", "tr", "cl", "cr"]

            ## send to the web app
            if self._ws_server.ws_camera_stream_on:
                try:
                    for x, img in enumerate(img_arr): # <-- Added parentheses around img_arr
                        _, img_buf = cv2.imencode('.jpg', img)
                        img_buf = base64.b64encode(img_buf)
                        img_buf = img_head_tag[x].encode() + img_buf
                        self._ws_server.wss_camera_feed_q.put_nowait(img_buf)
                
                except queue.Full:
                    pass

            if self.is_scanning is True:
                led.blink(0xffffff, 0.1, 2)
                self.save_capture_data()

                thermal_label_L, thermal_conf_L, thermal_prob_L = self.thermal_predictor.predict(thermal_camera_L.temp_raw_frame, True, self.current_scan_num_folder_path, "TL")
                thermal_label_R, thermal_conf_R, thermal_prob_R = self.thermal_predictor.predict(thermal_camera_R.temp_raw_frame, True, self.current_scan_num_folder_path, "TR")
                rgb_label_L, rgb_conf_L, rgb_prob_L = self.rgb_predictor.predict(rgb_img_L, True, self.current_scan_num_folder_path, "RGBL")
                rgb_label_R, rgb_conf_R, rgb_prob_R = self.rgb_predictor.predict(rgb_img_R, True, self.current_scan_num_folder_path, "RGBR")

                self.current_scan_max_thermal_conf_L = thermal_conf_L >= self.current_scan_max_thermal_conf_L if thermal_conf_L else self.current_scan_max_thermal_conf_L
                self.current_scan_max_thermal_conf_R = thermal_conf_R >= self.current_scan_max_thermal_conf_R if thermal_conf_R else self.current_scan_max_thermal_conf_R
                self.current_scan_max_rgb_conf_L = rgb_conf_L >= self.current_scan_max_rgb_conf_L if rgb_conf_L else self.current_scan_max_rgb_conf_L
                self.current_scan_max_rgb_conf_R = rgb_conf_R >= self.current_scan_max_rgb_conf_R if rgb_conf_R else self.current_scan_max_rgb_conf_R

                if self.current_scan_count >= global_settings["SCAN_SAVE_IMGS_NUM"]:
                    self._done_scan()

                time.sleep(global_settings["SCAN_TIME_INTERVAL_SEC"])

            time.sleep(0.05)

    def start_scan(self):
        ws_tx_str(json.dumps({"T":"STATUS", "DATA": "Scan Starting..."}))
        ws_tx_str(json.dumps({"T":"RESULT", "START": ""}))
        led.set_static_color(0xff0088)

        self.current_scan_folder_path = create_unique_folder(global_settings["save_images_base_path"], "patient")
        self.current_scan_count = 0
        self.current_scan_max_thermal_conf_L = 0.0
        self.current_scan_max_thermal_conf_R = 0.0
        self.current_scan_max_rgb_conf_L = 0.0
        self.current_scan_max_rgb_conf_R = 0.0

        time.sleep(1)
        self.is_scanning = True

    def _done_scan(self):
        self.is_scanning = False 

        if self.current_scan_max_thermal_conf_L >= global_settings["THERMAL_MODEL_THREAHOLD"] or self.current_scan_max_thermal_conf_R >= global_settings["THERMAL_MODEL_THREAHOLD"] or self.current_scan_max_rgb_conf_L >= global_settings["RGB_MODEL_THREAHOLD"] or self.current_scan_max_rgb_conf_R >= global_settings["RGB_MODEL_THREAHOLD"]:
            ws_tx_str(json.dumps({"T":"RESULT", "DATA": "DM / High Risk (Ulcer)"}))
            ws_tx_str(json.dumps({"T":"RESULT", "RISK": 0}))

        else:
            ws_tx_str(json.dumps({"T":"RESULT", "DATA": "Control / Healthy"}))
            ws_tx_str(json.dumps({"T":"RESULT", "RISK": 1}))

        ws_tx_str(json.dumps({"T":"STATUS", "DATA": "Scan Completed!"}))
        time.sleep(3)
        led.set_static_color(0x008000)
        ws_tx_str(json.dumps({"T":"STATUS", "DATA": "Ready!"}))

    def save_capture_data(self):
        self.current_scan_count += 1

        ws_tx_str(json.dumps({"T":"STATUS", "DATA": f"Capturing... {self.current_scan_count}"}))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.current_scan_folder_path, f"cap_{self.current_scan_count}_{timestamp}")
        os.makedirs(filepath, exist_ok=True)
        self.current_scan_num_folder_path = filepath

        thermal_camera_L.change_csv_save_path(filepath, filepath)
        thermal_camera_L.save_raw_csv()
        thermal_camera_L.save_temp_png()
        
        thermal_camera_R.change_csv_save_path(filepath, filepath)
        thermal_camera_R.save_raw_csv()
        thermal_camera_R.save_temp_png()

        rgb_cameraL.change_image_save_path(filepath, filepath)
        rgb_cameraL.save_image_png("L_")
        rgb_cameraL.save_intensity_matrix()
        
        rgb_cameraR.change_image_save_path(filepath, filepath)
        rgb_cameraR.save_image_png("R_")
        rgb_cameraR.save_intensity_matrix()



def controller_thread():

    while True:
        start_btn = push_btn.get_status(14)
        power_btn = push_btn.get_status(15)

        if start_btn == "SINGLE":
            led.blink(0x0000ff, 0.2, 1)
            image_proc_tsk_handler.start_scan()

        if power_btn == "LONG":
            led.blink(0xff0000, 1, 1)
            shutdown_raspberry_pi()
        elif power_btn == "DOUBLE":
            led.blink(0xff0000, 0.2, 2)
            restart_python_program()

        time.sleep(0.05)


def ws_tx_str(msg):
    try:
        server_data.wss_data_tx_q.put_nowait(msg)
    except queue.Full:
        pass

def ws_rx_data_handler(data):
    global server_data

    if data["T"] == "GET_SETTINGS":
        ws_tx_str(json.dumps(global_settings))

    elif data["T"] == "LIVE":
        if not server_data.ws_camera_stream_on:
            print("Openning Camera")
            cleanup_queue(server_data.wss_camera_feed_q)
            server_data.ws_camera_stream_on = True
        else:
            print("Closing Camera")
            server_data.ws_camera_stream_on = False

    elif data["T"] == "START":
        image_proc_tsk_handler.start_scan()

    elif data["T"] == "RESTART":
        restart_python_program()
        
    elif data["T"] == "POWER":
        shutdown_raspberry_pi()

    elif data["T"] == "SETTINGS":
        global_settings.update(data)
        json_cfg.save()
        print("saving json settings")


def cleanup_queue(x_q:queue.Queue):
    while not x_q.empty():
        try:
            x_q.get_nowait()
            x_q.task_done()  # Include this if you are using q.join() elsewhere
        except queue.Empty:
            break

async def ws_tx_hanlder_task(ws_t:SERVER_DATA_STRUCT):
    print("WS Tx Hanlder Started!")
    
    try:
        while True:
            if server_data.ws_client_connected and server_data.ws_client_handler is not None:
                try:
                    pkt = ws_t.wss_data_tx_q.get_nowait()
                    await server_data.ws_client_handler.send(pkt)
                    ws_t.wss_data_tx_q.task_done()
                    
                except queue.Empty:
                    pass

                if ws_t.ws_camera_stream_on is True:
                    try:
                        image_frame = ws_t.wss_camera_feed_q.get_nowait()
                        await server_data.ws_client_handler.send(image_frame)
                        ws_t.wss_camera_feed_q.task_done()
                        
                    except queue.Empty:
                        pass
            
            await asyncio.sleep(0.01)
    
    except asyncio.CancelledError:
        print("Cleaning WS Tx Task")
        cleanup_queue(server_data.wss_camera_feed_q)
        cleanup_queue(server_data.wss_data_tx_q)
        server_data.ws_camera_stream_on = False
        raise

    except Exception as e:
        print(f"WS task closed due to : {e}")

async def wshandler(websocket):
    global server_data
    server_data.ws_client_connected = True
    server_data.ws_client_handler = websocket
    
    try:
        async for message in websocket:
            try:
                ws_rx_data_handler(json.loads(message))
            except Exception as e:
                print(f"ERROR: WS rx data | {e}")
                
    except websockets.exceptions.ConnectionClosedOK:
        print("WS Client disconnected!")

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WS Client connection closed with error: {e}")

    finally:
        server_data.ws_client_connected = False
        server_data.ws_camera_stream_on = False
        server_data.ws_client_handler = None
        print("WS Connection handler finished.")

async def start_ws_server():
    async with websockets.serve(wshandler, "0.0.0.0", 8000):
        print(f"App server started @ {socket.gethostname()}.local")
        loop = asyncio.get_running_loop()
        server_data.ws_camera_stream_task = loop.create_task(ws_tx_hanlder_task(server_data))
        await asyncio.Future()

def cleanup_resources():
    """Stops threads and cancels async tasks before exiting."""
    print("Cleaning up threads and tasks...")
    led.blink(0x000080, 0.4)
    time.sleep(0.5)
    
    try:
        push_btn.cleanup()
        push_btn.join(timeout=3)
    except Exception as e:
        print(f"Error stopping led thread: {e}")
    
    # Cancel the camera stream task
    try:
        if hasattr(server_data, 'ws_camera_stream_task') and server_data.ws_camera_stream_task:
            server_data.ws_camera_stream_task.cancel()
    except Exception as e:
        print(f"Error canceling camera task: {e}")

    # Stop the image processing thread
    try:
        if hasattr(image_proc_tsk_handler, 'stop'):
            image_proc_tsk_handler.stop()
            image_proc_tsk_handler.join(timeout=3) # 3-second timeout prevents freezing
    except Exception as e:
        print(f"Error stopping image thread: {e}")

    try:
        led.cleanup()
        led.join(timeout=3)
    except Exception as e:
        print(f"Error stopping led thread: {e}")

def shutdown_raspberry_pi():
    """Cleans up the app and completely powers off the Raspberry Pi."""
    print("Initiating system shutdown...")
    cleanup_resources()
    
    # 'sudo poweroff' is the standard way to shut down a Raspberry Pi securely
    os.system("sudo poweroff")

def restart_python_program():
    """Cleans up the app and restarts the Python script within the same venv."""
    print("Restarting Python program...")
    cleanup_resources()
    
    # Force-close all inherited file descriptors/sockets (ports 5000, 8000, etc.)
    # 3 is the first non-standard descriptor. 65535 covers the maximum limit.
    try:
        os.closerange(3, 65535)
    except Exception as e:
        print(f"Error closing ports: {e}")
    
    # sys.executable ensures it uses the exact same venv Python binary
    # sys.argv passes along any command line arguments you originally used
    os.execv(sys.executable, [sys.executable] + sys.argv)

def check_app_ready():
    while not get_server_status():
        time.sleep(1)

    led.set_static_color(0x008000)

# Dictionary to keep the pin connections alive in memory
_active_pins = {}

def set_gpio(pin_num, turn_on):
    """
    Turns a GPIO pin ON or OFF.
    Usage: set_gpio(14, True) to turn ON, set_gpio(14, False) to turn OFF.
    """
    # If this is the first time using the pin, initialize it
    if pin_num not in _active_pins:
        _active_pins[pin_num] = DigitalOutputDevice(pin_num)
    
    # Set the state
    if turn_on:
        _active_pins[pin_num].on()
    else:
        _active_pins[pin_num].off()

if __name__ == "__main__":
    time.sleep(0.5)
    check_for_updates_and_restart()
    print("Starting Medical App...")

    led = WS2812BThread(pin_num=18, num_leds=2)
    led.set_static_color(0xFFFF00)
    led.start()
    time.sleep(0.5)
    set_gpio(4, True)
    
    json_cfg = JsonConfigManager(global_settings, "saved_settings.json")
    json_cfg.load_and_sync()

    push_btn = ButtonMonitor()
    image_proc_tsk_handler = ImageProcessWorker(server_data)
    flask_app_thread = threading.Thread(target=run_web_app, daemon=True)
    device_controller_th = threading.Thread(target=controller_thread, daemon=True)

    push_btn.add_button(pin=14, pull_state="UP")
    push_btn.add_button(pin=15, pull_state="UP")

    push_btn.start()
    device_controller_th.start()
    image_proc_tsk_handler.start()
    flask_app_thread.start()
    
    check_app_ready_th = threading.Thread(target=check_app_ready, daemon=True)
    check_app_ready_th.start()

    print("App Started!")

    try:
        asyncio.run(start_ws_server())
    
    except KeyboardInterrupt:
        print("Closing App")

        cleanup_resources()
        exit()


