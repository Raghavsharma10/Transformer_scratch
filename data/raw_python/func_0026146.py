def setGyroSensitivity(self, value):
        """
        Sets the gyro sensitivity to 250, 500, 1000 or 2000 according to the given value (and implicitly disables the
        self
        tests)
        :param value: the target sensitivity.
        """
        try:
            self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_GYRO_CONFIG,
                              {250: 0, 500: 8, 1000: 16, 2000: 24}[value])
            self._gyroFactor = value / 32768.0
            self.gyroSensitivity = value
            logger.debug("Set gyro sensitivity = %d", value)
        except KeyError:
            raise ArgumentError(value + " is not a valid sensitivity (250,500,1000,2000)")