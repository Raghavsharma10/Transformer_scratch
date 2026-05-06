def read(self):
        """Read the current value of the accelerometer and return it as a tuple
        of signed 16-bit X, Y, Z axis values.
        """
        raw = self._device.readList(ADXL345_REG_DATAX0, 6)
        return struct.unpack('<hhh', raw)