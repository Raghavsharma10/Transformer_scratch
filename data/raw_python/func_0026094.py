def get(self, deviceId, measurementId):
        """
        details the specific measurement.
        """
        record = self.measurements.get(deviceId)
        if record is not None:
            return record.get(measurementId)
        return None