def resetFifo(self):
        """
        Resets the FIFO by first disabling the FIFO then sending a FIFO_RESET and then re-enabling the FIFO.
        :return:
        """
        logger.debug("Resetting FIFO")
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_USER_CTRL, 0b00000000)
        pass
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_USER_CTRL, 0b00000100)
        pass
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_USER_CTRL, 0b01000000)
        self.getInterruptStatus()