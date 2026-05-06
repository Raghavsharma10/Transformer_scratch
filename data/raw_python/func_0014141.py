def enable(self):
        """Enable the chip."""
        # Flip on the power and enable bits.
        self._write8(TCS34725_ENABLE, TCS34725_ENABLE_PON)
        time.sleep(0.01)
        self._write8(TCS34725_ENABLE, (TCS34725_ENABLE_PON | TCS34725_ENABLE_AEN))