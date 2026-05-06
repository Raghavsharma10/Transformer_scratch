def put(self, deviceId, measurementId):
        """
        Schedules a new measurement at the specified time.
        :param deviceId: the device to measure.
        :param measurementId: the name of the measurement.
        :return: 200 if it was scheduled, 400 if the device is busy, 500 if the device is bad.
        """
        record = self.measurements.get(deviceId)
        if record is not None:
            measurement = record.get(measurementId)
            if measurement is not None:
                if len([x.name for x in measurement.statuses if x.name is 'COMPLETE' or x.name is 'FAILED']) > 0:
                    logger.info('Overwriting existing completed measurement ' + x.name)
                    measurement = None
            if measurement is None:
                logger.info('Initiating measurement ' + measurementId)
                measurement = ScheduledMeasurement(measurementId, self.recordingDevices.get(deviceId))
                body = request.get_json()
                duration_ = body['duration']
                def _cleanup():
                    logger.info('Removing ' + measurementId + ' from ' + deviceId)
                    record.pop(measurementId)
                measurement.schedule(duration_, at=body.get('at'), delay=body.get('delay'), callback=_cleanup)
                # a quick hack to enable the measurement to be cleaned up by the ScheduledMeasurement
                record[measurementId] = measurement
                return measurement, 200
            else:
                return measurement, 400
        else:
            return 'unknown device ' + deviceId, 400