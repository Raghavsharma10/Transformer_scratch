def enableAccelerometer(self):
        """
        Specifies the device should write acceleration values to the FIFO, is not applied until enableFIFO is called.
        :return:
        """
        logger.debug("Enabling acceleration sensor")
        self.fifoSensorMask |= self.enableAccelerometerMask
        self._accelEnabled = True
        self._setSampleSizeBytes()