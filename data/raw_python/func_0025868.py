def _deleteCompletedMeasurement(self, measurementId):
        """
        Deletes the named measurement from the completed measurement store if it exists.
        :param measurementId:
        :return:
            String: error messages
            Integer: count of measurements deleted
        """
        message, count, deleted = self.deleteFrom(measurementId, self.completeMeasurements)
        if count is 0:
            message, count, deleted = self.deleteFrom(measurementId, self.failedMeasurements)
        return message, count, deleted