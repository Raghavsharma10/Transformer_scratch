def recordData(self, measurementId, deviceId, data):
        """
        Passes the data to the handler.
        :param deviceId: the device the data comes from.
        :param measurementId: the measurement id.
        :param data: the data.
        :return: true if the data was handled.
        """
        am, handler = self.getDataHandler(measurementId, deviceId)
        if handler is not None:
            am.stillRecording(deviceId, len(data))
            handler.handle(data)
            return True
        else:
            logger.error('Received data for unknown handler ' + deviceId + '/' + measurementId)
            return False