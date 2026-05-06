def put(self, measurementId, deviceId):
        """
        Initialises the measurement session from the given device.
        :param measurementId:
        :param deviceId:
        :return:
        """
        logger.info('Starting measurement ' + measurementId + ' for ' + deviceId)
        if self._measurementController.startMeasurement(measurementId, deviceId):
            logger.info('Started measurement ' + measurementId + ' for ' + deviceId)
            return None, 200
        else:
            logger.warning('Failed to start measurement ' + measurementId + ' for ' + deviceId)
            return None, 404