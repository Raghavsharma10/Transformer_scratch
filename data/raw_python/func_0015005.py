def set_thresholds(self, touch, release):
        """Set the touch and release threshold for all inputs to the provided
        values.  Both touch and release should be a value between 0 to 255
        (inclusive).
        """
        assert touch >= 0 and touch <= 255, 'touch must be between 0-255 (inclusive)'
        assert release >= 0 and release <= 255, 'release must be between 0-255 (inclusive)'
        # Set the touch and release register value for all the inputs.
        for i in range(12):
            self._i2c_retry(self._device.write8, MPR121_TOUCHTH_0 + 2*i, touch)
            self._i2c_retry(self._device.write8, MPR121_RELEASETH_0 + 2*i, release)