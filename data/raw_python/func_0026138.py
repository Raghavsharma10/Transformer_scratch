def getPacketSize(self):
        """
        the current packet size.
        :return: the current packet size based on the enabled registers.
        """
        size = 0
        if self.isAccelerometerEnabled():
            size += 6
        if self.isGyroEnabled():
            size += 6
        if self.isTemperatureEnabled():
            size += 2
        return size