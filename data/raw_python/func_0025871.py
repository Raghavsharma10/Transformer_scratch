def getMeasurement(self, measurementId, measurementStatus=None):
        """
        Gets the measurement with the given id.
        :param measurementId: the id.
        :param measurementStatus: the status of the requested measurement.
        :return: the matching measurement or none if it doesn't exist.
        """
        return next((x for x in self.getMeasurements(measurementStatus) if x.id == measurementId), None)