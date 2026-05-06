def delete(self, measurementId):
        """
        Deletes the named measurement.
        :return: 200 if something was deleted, 404 if the measurement doesn't exist, 500 in any other case.
        """
        message, count, deleted = self._measurementController.delete(measurementId)
        if count == 0:
            return message, 404
        elif deleted is None:
            return message, 500
        else:
            return deleted, 200