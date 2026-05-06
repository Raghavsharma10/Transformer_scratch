def delete(self, deviceId, measurementId):
        """
        Deletes a stored measurement.
        :param deviceId: the device to measure.
        :param measurementId: the name of the measurement.
        :return: 200 if it was deleted, 400 if no such measurement (or device).
        """
        record = self.measurements.get(deviceId)
        if record is not None:
            popped = record.pop(measurementId, None)
            return popped, 200 if popped else 400
        return None, 400