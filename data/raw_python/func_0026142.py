def enableGyro(self):
        """
        Specifies the device should write gyro values to the FIFO, is not applied until enableFIFO is called.
        :return: 
        """
        logger.debug("Enabling gyro sensor")
        self.fifoSensorMask |= self.enableGyroMask
        self._gyroEnabled = True
        self._setSampleSizeBytes()