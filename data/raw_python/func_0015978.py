def getDescriptor(self):
        """
        Returns the currently active endpoint descriptor
        (depending on current USB speed).
        """
        result = USBEndpointDescriptor()
        self._ioctl(ENDPOINT_DESC, result, True)
        return result