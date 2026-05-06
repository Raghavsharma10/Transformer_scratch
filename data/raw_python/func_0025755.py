def start(self, measurementId):
        """
        Posts to the target to tell it a named measurement is starting.
        :param measurementId:
        """
        self.sendURL = self.rootURL + measurementId + '/' + self.deviceName
        self.startResponseCode = self._doPut(self.sendURL)