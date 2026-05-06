def put(self, measurementId, deviceId):
        """
        Store a bunch of data for this measurement session.
        :param measurementId:
        :param deviceId:
        :return:
        """
        data = request.get_json()
        if data is not None:
            parsedData = json.loads(data)
            logger.debug('Received payload ' + measurementId + '/' + deviceId + ': ' +
                         str(len(parsedData)) + ' records')
            if self._measurementController.recordData(measurementId, deviceId, parsedData):
                return None, 200
            else:
                logger.warning('Unable to record payload ' + measurementId + '/' + deviceId)
                return None, 404
        else:
            logger.error('Invalid data payload received ' + measurementId + '/' + deviceId)
            return None, 400