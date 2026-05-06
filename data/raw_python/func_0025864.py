def getDataHandler(self, measurementId, deviceId):
        """
        finds the handler.
        :param measurementId: the measurement
        :param deviceId: the device.
        :return: active measurement and handler
        """
        am = next((m for m in self.activeMeasurements if m.id == measurementId), None)
        if am is None:
            return None, None
        else:
            device = self.deviceController.getDevice(deviceId)
            if device is None:
                return None, None
            else:
                return am, device.dataHandler