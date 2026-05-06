def disableGyro(self):
        """
        Specifies the device should NOT write gyro values to the FIFO, is not applied until enableFIFO is called.
        :return: 
        """
        logger.debug("Disabling gyro sensor")
        self.fifoSensorMask &= ~self.enableGyroMask
        self._gyroEnabled = False
        self._setSampleSizeBytes()