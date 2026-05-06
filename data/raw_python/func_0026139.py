def initialiseDevice(self):
        """
        performs initialisation of the device
        :param batchSize: the no of samples that each provideData call should yield
        :return:
        """
        logger.debug("Initialising device")
        self.getInterruptStatus()
        self.setAccelerometerSensitivity(self._accelerationFactor * 32768.0)
        self.setGyroSensitivity(self._gyroFactor * 32768.0)
        self.setSampleRate(self.fs)
        for loop in self.ZeroRegister:
            self.i2c_io.write(self.MPU6050_ADDRESS, loop, 0)
        # Sets clock source to gyro reference w/ PLL
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_PWR_MGMT_1, 0b00000010)
        # Controls frequency of wakeups in accel low power mode plus the sensor standby modes
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_PWR_MGMT_2, 0x00)
        # Enables any I2C master interrupt source to generate an interrupt
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_INT_ENABLE, 0x01)
        # enable the FIFO
        self.enableFifo()
        logger.debug("Initialised device")