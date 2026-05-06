def get(self, deviceId, measurementId):
        """
        signals a stop for the given measurement.
        :param deviceId: the device to measure.
        :param measurementId: the name of the measurement.
        :return: 200 if stop is signalled, 400 if it doesn't exist or is not running.
        """
        record = self.measurements.get(deviceId)
        if record is not None:
            measurement = record.get(measurementId)
            if measurement.recording:
                device = self.recordingDevices.get(deviceId)
                device.signalStop()
                return measurement, 200
            else:
                return measurement, 400
        return '', 400