import time
import threading
from gpiozero import Button

class ButtonMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Dictionary to store the gpiozero Button objects and their states
        self.buttons = {}
        
        # Timing thresholds (in seconds)
        self.DOUBLE_CLICK_TIME = 0.35
        self.LONG_PRESS_TIME = 1.0

    def add_button(self, pin, pull_state="UP"):
        """
        Adds a button using gpiozero.
        pull_state: "UP" (connects button to GND) or "DOWN" (connects button to 3.3V)
        """
        is_pull_up = True if pull_state.upper() == "UP" else False
        
        # gpiozero automatically handles debouncing (bounce_time=0.05 is 50ms)
        btn = Button(pin, pull_up=is_pull_up, bounce_time=0.05)
        
        with self._lock:
            self.buttons[pin] = {
                'hardware': btn,
                'is_pressed': False,
                'press_start': 0.0,
                'last_release': 0.0,
                'clicks': 0,
                'long_triggered': False,
                'current_event': 'NONE'
            }
        print(f"[ButtonMonitor] Tracking Pin {pin} | Pull: {pull_state}")

    def get_status(self, pin):
        """
        Reads the button status. 
        Returns: 'NONE', 'SINGLE', 'DOUBLE', or 'LONG'.
        """
        with self._lock:
            if pin not in self.buttons:
                return 'NONE'
            
            event = self.buttons[pin]['current_event']
            self.buttons[pin]['current_event'] = 'NONE'
            return event

    def cleanup(self):
        """Closes all hardware connections cleanly."""
        self._stop_event.set()
        if self.is_alive():
            self.join()
            
        with self._lock:
            for pin, state in self.buttons.items():
                state['hardware'].close()
        print("[ButtonMonitor] Hardware cleaned up.")

    def run(self):
        """Main background loop."""
        while not self._stop_event.is_set():
            now = time.time()
            
            with self._lock:
                for pin, state in self.buttons.items():
                    btn = state['hardware']
                    is_pressed_now = btn.is_pressed
                    
                    # 1. BUTTON JUST PRESSED DOWN
                    if is_pressed_now and not state['is_pressed']:
                        state['is_pressed'] = True
                        state['press_start'] = now
                        state['long_triggered'] = False
                        
                    # 2. BUTTON IS BEING HELD DOWN
                    elif is_pressed_now and state['is_pressed']:
                        held_time = now - state['press_start']
                        if held_time >= self.LONG_PRESS_TIME and not state['long_triggered']:
                            state['current_event'] = 'LONG'
                            state['long_triggered'] = True
                            state['clicks'] = 0 
                            
                    # 3. BUTTON JUST RELEASED
                    elif not is_pressed_now and state['is_pressed']:
                        state['is_pressed'] = False
                        if not state['long_triggered']:
                            state['clicks'] += 1
                            state['last_release'] = now
                            
                    # 4. BUTTON IS IDLE 
                    elif not is_pressed_now and not state['is_pressed']:
                        if state['clicks'] > 0 and (now - state['last_release'] > self.DOUBLE_CLICK_TIME):
                            if state['clicks'] == 1:
                                state['current_event'] = 'SINGLE'
                            elif state['clicks'] >= 2:
                                state['current_event'] = 'DOUBLE'
                            
                            state['clicks'] = 0

            time.sleep(0.01)