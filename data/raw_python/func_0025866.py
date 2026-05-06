def completeMeasurement(self, measurementId, deviceId):
        """
        Completes the measurement session.
        :param deviceId: the device id.
        :param measurementId: the measurement id.
        :return: true if it was completed.
        """
        am, handler = self.getDataHandler(measurementId, deviceId)
        if handler is not None:
            handler.stop(measurementId)
            am.updateDeviceStatus(deviceId, RecordStatus.COMPLETE)
            return True
        else:
            return False