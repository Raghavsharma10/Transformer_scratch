def failMeasurement(self, measurementId, deviceName, failureReason=None):
        """
        Fails the measurement session.
        :param deviceName: the device name.
        :param measurementId: the measurement name.
        :param failureReason: why it failed.
        :return: true if it was completed.
        """
        am, handler = self.getDataHandler(measurementId, deviceName)
        if handler is not None:
            am.updateDeviceStatus(deviceName, RecordStatus.FAILED, reason=failureReason)
            handler.stop(measurementId)
            return True
        else:
            return False