def get(self, deviceId):
        """
        lists all known active measurements.
        """
        measurementsByName = self.measurements.get(deviceId)
        if measurementsByName is None:
            return []
        else:
            return list(measurementsByName.values())