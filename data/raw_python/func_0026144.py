def enableTemperature(self):
        """
        Specifies the device should write temperature values to the FIFO, is not applied until enableFIFO is called.
        :return: 
        """
        logger.debug("Enabling temperature sensor")
        self.fifoSensorMask |= self.enableTemperatureMask
        self._setSampleSizeBytes()