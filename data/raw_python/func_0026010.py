def scheduleMeasurement(self, measurementId, duration, start):
        """
        Schedules the requested measurement session with all INITIALISED devices.
        :param measurementId:
        :param duration:
        :param start:
        :return: a dict of device vs status.
        """
        # TODO subtract 1s from start and format
        results = {}
        for device in self.getDevices(RecordingDeviceStatus.INITIALISED.name):
            logger.info('Sending measurement ' + measurementId + ' to ' + device.payload['serviceURL'])
            try:
                resp = self.httpclient.put(device.payload['serviceURL'] + '/measurements/' + measurementId,
                                           json={'duration': duration, 'at': start.strftime(DATETIME_FORMAT)})
                logger.info('Response for ' + measurementId + ' from ' + device.payload['serviceURL'] + ' is ' +
                            str(resp.status_code))
                results[device] = resp.status_code
            except Exception as e:
                logger.exception(e)
                results[device] = 500
        return results