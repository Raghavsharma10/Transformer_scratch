def _write8(self, reg, value):
        """Write a 8-bit value to a register."""
        self._device.write8(TCS34725_COMMAND_BIT | reg, value)