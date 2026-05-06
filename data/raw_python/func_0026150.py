def enableFifo(self):
        """
        Enables the FIFO, resets it and then sets which values should be written to the FIFO.
        :return:
        """
        logger.debug("Enabling FIFO")
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_FIFO_EN, 0)
        self.resetFifo()
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_FIFO_EN, self.fifoSensorMask)
        logger.debug("Enabled FIFO")