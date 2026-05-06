def disable(self):
        """Disable the chip (power down)."""
        # Flip off the power on and enable bits.
        reg = self._readU8(TCS34725_ENABLE)
        reg &= ~(TCS34725_ENABLE_PON | TCS34725_ENABLE_AEN)
        self._write8(TCS34725_ENABLE, reg)