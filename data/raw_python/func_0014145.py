def set_interrupt(self, enabled):
        """Enable or disable interrupts by setting enabled to True or False."""
        enable_reg = self._readU8(TCS34725_ENABLE)
        if enabled:
            enable_reg |= TCS34725_ENABLE_AIEN
        else:
            enable_reg &= ~TCS34725_ENABLE_AIEN
        self._write8(TCS34725_ENABLE, enable_reg)
        time.sleep(1)