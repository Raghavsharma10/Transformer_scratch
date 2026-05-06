def startMeasurement(self, measurementId, deviceId):
        """
        Starts the measurement for the device.
        :param deviceId: the device that is starting.
        :param measurementId: the measurement that is started.
        :return: true if it started (i.e. device and measurement exists).
        """
        am, handler = self.getDataHandler(measurementId, deviceId)
        if am is not None:
            am.status = MeasurementStatus.RECORDING
            am.updateDeviceStatus(deviceId, RecordStatus.RECORDING)
            handler.start(am.idAsPath)
            return True
        else:
            return False