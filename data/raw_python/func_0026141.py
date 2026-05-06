def disableAccelerometer(self):
        """
        Specifies the device should NOT write acceleration values to the FIFO, is not applied until enableFIFO is
        called.
        :return: 
        """
        logger.debug("Disabling acceleration sensor")
        self.fifoSensorMask &= ~self.enableAccelerometerMask
        self._accelEnabled = False
        self._setSampleSizeBytes()