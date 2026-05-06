def setAccelerometerSensitivity(self, value):
        """
        Sets the accelerometer sensitivity to 2, 4, 8 or 16 according to the given value. Throws an ArgumentError if
        the value provided is not valid.
        :param value: the target sensitivity.
        """
        # note that this implicitly disables the self tests on each axis
        # i.e. the full byte is actually 000[accel]000 where the 1st 3 are the accelerometer self tests, the next two
        # values are the actual sensitivity and the last 3 are unused
        # the 2 [accel] bits are translated by the device as follows; 00 = 2g, 01 = 4g, 10 = 8g, 11 = 16g
        # in binary we get 2 = 0, 4 = 1000, 8 = 10000, 16 = 11000
        # so the 1st 3 bits are always 0
        try:
            self.i2c_io.write(self.MPU6050_ADDRESS,
                              self.MPU6050_RA_ACCEL_CONFIG,
                              {2: 0, 4: 8, 8: 16, 16: 24}[value])
            self._accelerationFactor = value / 32768.0
            self.accelerometerSensitivity = value
            logger.debug("Set accelerometer sensitivity = %d", value)
        except KeyError:
            raise ArgumentError(value + " is not a valid sensitivity (2,4,8,18)")