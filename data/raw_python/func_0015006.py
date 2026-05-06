def filtered_data(self, pin):
        """Return filtered data register value for the provided pin (0-11).
        Useful for debugging.
        """
        assert pin >= 0 and pin < 12, 'pin must be between 0-11 (inclusive)'
        return self._i2c_retry(self._device.readU16LE, MPR121_FILTDATA_0L + pin*2)