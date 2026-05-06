def set_interrupt_limits(self, low, high):
        """Set the interrupt limits to provied unsigned 16-bit threshold values.
        """
        self._device.write8(0x04, low & 0xFF)
        self._device.write8(0x05, low >> 8)
        self._device.write8(0x06, high & 0xFF)
        self._device.write8(0x07, high >> 8)