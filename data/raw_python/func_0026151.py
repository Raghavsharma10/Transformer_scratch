def getFifoCount(self):
        """
        gets the amount of data available on the FIFO right now.
        :return: the number of bytes available on the FIFO which will be proportional to the number of samples available
        based on the values the device is configured to sample.
        """
        bytes = self.i2c_io.readBlock(self.MPU6050_ADDRESS, self.MPU6050_RA_FIFO_COUNTH, 2)
        count = (bytes[0] << 8) + bytes[1]
        logger.debug("FIFO Count: %d", count)
        return count