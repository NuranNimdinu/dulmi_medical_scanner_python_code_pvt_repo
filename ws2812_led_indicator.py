import time
import threading
import board
import neopixel

class WS2812BThread(threading.Thread):
    def __init__(self, pin_num, num_leds):
        super().__init__(daemon=True)
        self.num_leds = num_leds
        
        # Dynamically get the board pin (e.g., pin_num=18 maps to board.D18)
        pixel_pin = getattr(board, f"D{pin_num}")
        
        # Initialize the NeoPixel strip
        self.pixels = neopixel.NeoPixel(
            pixel_pin, num_leds, brightness=1.0, auto_write=False, pixel_order=neopixel.GRB
        )
        
        # Thread control flags
        self._stop_event = threading.Event()   # Kill thread flag
        self._update_event = threading.Event() # Interrupt/wake-up flag
        self._mode_lock = threading.Lock()     # Mutex lock for thread safety
        
        # State variables
        self._mode = 'CLEAR'
        self._current_color = (0, 0, 0)
        self._previous_static_color = None     # Tracks the baseline static color
        self._interval = 0.5
        self._pulses_left = 0
        self._infinite_blink = False

    def _hex_to_rgb(self, hex_color):
        """Converts an integer hex code (e.g., 0xFF0000) to an RGB tuple (255, 0, 0)."""
        r = (hex_color >> 16) & 0xFF
        g = (hex_color >> 8) & 0xFF
        b = hex_color & 0xFF
        return (r, g, b)

    def set_static_color(self, hex_color):
        """Sets the LEDs to a solid color and saves it as the baseline state."""
        with self._mode_lock:
            self._mode = 'STATIC'
            rgb = self._hex_to_rgb(hex_color)
            self._current_color = rgb
            self._previous_static_color = rgb
        self._update_event.set()

    def blink(self, hex_color, interval, pulses=0):
        """
        Blinks the LEDs. 
        interval: duration in seconds for half-cycle (on time = interval, off time = interval).
        pulses: Number of blinks. 0 or None means infinite.
        """
        with self._mode_lock:
            self._mode = 'BLINK'
            self._current_color = self._hex_to_rgb(hex_color)
            self._interval = interval
            self._pulses_left = pulses
            self._infinite_blink = (pulses <= 0 or pulses is None)
        self._update_event.set()

    def stop_blink(self):
        """
        Stops blinking immediately.
        Restores previous static color if one was active, otherwise turns LEDs off.
        """
        with self._mode_lock:
            if self._previous_static_color is not None:
                self._mode = 'STATIC'
                self._current_color = self._previous_static_color
            else:
                self._mode = 'CLEAR'
                self._current_color = (0, 0, 0)
        self._update_event.set()

    def clear(self):
        """Turns off all LEDs and clears any remembered static color."""
        with self._mode_lock:
            self._mode = 'CLEAR'
            self._current_color = (0, 0, 0)
            self._previous_static_color = None
        self._update_event.set()

    def cleanup(self):
        """Safely stops the thread and clears hardware."""
        self._stop_event.set()
        self._update_event.set()
        
        if self.is_alive():
            self.join()
            
        self.pixels.fill((0, 0, 0))
        self.pixels.show()
        self.pixels.deinit()

    def run(self):
        """Main background loop."""
        while not self._stop_event.is_set():
            with self._mode_lock:
                mode = self._mode
                color = self._current_color
                interval = self._interval
                infinite = self._infinite_blink
            
            self._update_event.clear()
            
            if mode == 'CLEAR':
                self.pixels.fill((0, 0, 0))
                self.pixels.show()
                self._update_event.wait()
                
            elif mode == 'STATIC':
                self.pixels.fill(color)
                self.pixels.show()
                self._update_event.wait()
                
            elif mode == 'BLINK':
                # ON Phase
                self.pixels.fill(color)
                self.pixels.show()
                if self._update_event.wait(timeout=interval):
                    continue 
                
                # OFF Phase
                self.pixels.fill((0, 0, 0))
                self.pixels.show()
                if self._update_event.wait(timeout=interval):
                    continue
                    
                # Decrement pulses after complete ON/OFF cycle
                if not infinite:
                    with self._mode_lock:
                        if self._mode == 'BLINK':
                            self._pulses_left -= 1
                            if self._pulses_left <= 0:
                                # Revert automatically to previous static color or clear
                                if self._previous_static_color is not None:
                                    self._mode = 'STATIC'
                                    self._current_color = self._previous_static_color
                                else:
                                    self._mode = 'CLEAR'
                                    self._current_color = (0, 0, 0)