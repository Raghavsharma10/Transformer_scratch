def touched(self):
        """Return touch state of all pins as a 12-bit value where each bit 
        represents a pin, with a value of 1 being touched and 0 not being touched.
        """
        t = self._i2c_retry(self._device.readU16LE, MPR121_TOUCHSTATUS_L)
        return t & 0x0FFF