def getDevice(self, id):
        """
        gets the named device.
        :param id: the id.
        :return: the device
        """
        return next(iter([d for d in self.devices.values() if d.deviceId == id]), None)