def get_raw_data(self):
        """Reads the raw red, green, blue and clear channel values. Will return
        a 4-tuple with the red, green, blue, clear color values (unsigned 16-bit
        numbers).
        """
        # Read each color register.
        r = self._readU16LE(TCS34725_RDATAL)
        g = self._readU16LE(TCS34725_GDATAL)
        b = self._readU16LE(TCS34725_BDATAL)
        c = self._readU16LE(TCS34725_CDATAL)
        # Delay for the integration time to allow for next reading immediately.
        time.sleep(INTEGRATION_TIME_DELAY[self._integration_time])
        return (r, g, b, c)