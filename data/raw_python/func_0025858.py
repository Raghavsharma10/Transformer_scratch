def stillRecording(self, deviceId, dataCount):
        """
        For a device that is recording, updates the last timestamp so we now when we last received data.
        :param deviceId: the device id.
        :param dataCount: the no of items of data recorded in this batch.
        :return:
        """
        status = self.recordingDevices[deviceId]
        if status is not None:
            if status['state'] == MeasurementStatus.RECORDING.name:
                status['last'] = datetime.datetime.utcnow().strftime(DATETIME_FORMAT)
                status['count'] = status['count'] + dataCount