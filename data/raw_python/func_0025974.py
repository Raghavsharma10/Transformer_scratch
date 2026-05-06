def put(self, measurementId, deviceId):
        """
        Fails the measurement for this device.
        :param measurementId: the measurement name.
        :param deviceId: the device name.
        :return: 200 if
        """
        payload = request.get_json()
        failureReason = json.loads(payload).get('failureReason') if payload is not None else None
        logger.warning('Failing measurement ' + measurementId + ' for ' + deviceId + ' because ' + str(failureReason))
        if self._measurementController.failMeasurement(measurementId, deviceId, failureReason=failureReason):
            logger.warning('Failed measurement ' + measurementId + ' for ' + deviceId)
            return None, 200
        else:
            logger.error('Unable to fail measurement ' + measurementId + ' for ' + deviceId)
            return None, 404