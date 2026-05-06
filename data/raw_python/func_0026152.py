def getDataFromFIFO(self, bytesToRead):
        """
        reads the specified number of bytes from the FIFO, should be called after a call to getFifoCount to ensure there
        is new data available (to avoid reading duplicate data).
        :param bytesToRead: the number of bytes to read.
        :return: the bytes read.
        """
        return self.i2c_io.readBlock(self.MPU6050_ADDRESS, self.MPU6050_RA_FIFO_R_W, bytesToRead)