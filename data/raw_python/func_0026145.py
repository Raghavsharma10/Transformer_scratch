def disableTemperature(self):
        """
        Specifies the device should NOT write temperature values to the FIFO, is not applied until enableFIFO is called.
        :return: 
        """
        logger.debug("Disabling temperature sensor")
        self.fifoSensorMask &= ~self.enableTemperatureMask
        self._setSampleSizeBytes()