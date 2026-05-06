def baseline_data(self, pin):
        """Return baseline data register value for the provided pin (0-11).
        Useful for debugging.
        """
        assert pin >= 0 and pin < 12, 'pin must be between 0-11 (inclusive)'
        bl = self._i2c_retry(self._device.readU8, MPR121_BASELINE_0 + pin)
        return bl << 2